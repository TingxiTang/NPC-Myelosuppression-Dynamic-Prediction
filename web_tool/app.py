"""Streamlit research-use interface for the frozen three-endpoint models."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from web_runtime import (
    ArtifactValidationError,
    FrozenWebPredictor,
    InputValidationError,
    build_pdf_report,
    prepare_uploaded_frame,
    synthetic_frame,
)
from web_runtime.input_data import frame_to_csv


LOGGER = logging.getLogger(__name__)
WEB_ROOT = Path(__file__).resolve().parent
ENDPOINTS = ("hb", "plt", "wbc_neut")
ENDPOINT_META = {
    "hb": {"label": "Hb", "zh": "Grade ≥3 贫血", "en": "Grade ≥3 anemia"},
    "plt": {"label": "PLT", "zh": "Grade ≥3 血小板减少", "en": "Grade ≥3 thrombocytopenia"},
    "wbc_neut": {
        "label": "WBC/Neut",
        "zh": "Grade ≥3 白细胞/中性粒细胞减少",
        "en": "Grade ≥3 leukopenia/neutropenia",
    },
}
TEXT = {
    "zh": {
        "page_title": "骨髓抑制风险预测系统",
        "kicker": "骨髓抑制风险预测系统",
        "title": "治疗周期血液学毒性风险预测",
        "subtitle": "上传一个治疗周期或使用演示样例，即可查看三项 Grade ≥3 血液学毒性风险及 SHAP 解释。",
        "notice_title": "研究用途提示",
        "notice": "本工具未经临床影响评估，不用于诊断、治疗选择、剂量调整或替代临床判断。",
        "input_step": "1. 输入治疗周期",
        "sample": "演示样例",
        "upload": "上传 CSV",
        "sample_label": "选择演示样例",
        "neutral": "中性合成样例",
        "lower_reserve": "较低血液学储备样例",
        "sample_note": "演示样例为无身份合成数据，不代表典型临床场景。",
        "template": "下载演示 CSV",
        "upload_label": "上传单行 CSV",
        "upload_help": "仅接收一个治疗周期；身份、日期及非模型字段会被拒绝。",
        "analyze": "开始分析",
        "running": "正在计算三项风险与 SHAP 解释…",
        "input_error": "输入未通过校验。",
        "prediction_error": "预测或 SHAP 校验失败，已停止输出。",
        "results_step": "2. 风险预测结果",
        "threshold": "阈值",
        "high": "达到警报阈值",
        "low": "未达到警报阈值",
        "explain_step": "3. SHAP 风险解释",
        "endpoint_select": "选择要解释的结局",
        "explain_note": "红色提高预测风险，蓝色降低预测风险；SHAP 表示预测贡献，不代表治疗因果效应。",
        "waterfall_tab": "SHAP 瀑布图",
        "overview_tab": "贡献概览",
        "waterfall_title": "从模型基线到本次预测的累计贡献",
        "overview_title": "主要特征贡献方向",
        "waterfall_xaxis": "锁定校准 logit",
        "contribution_xaxis": "锁定校准 logit 贡献",
        "other": "其余特征",
        "output": "本次预测",
        "report_step": "4. 下载报告",
        "report_note": "报告仅包含三项风险结果和 SHAP 贡献，不包含姓名、ID、日期或完整输入行。",
        "report_button": "下载中文 PDF 报告",
        "footer": "研究用途原型 · 风险警报不是临床建议",
        "asset_error": "模型资源校验失败，已禁止预测。",
    },
    "en": {
        "page_title": "Bone Marrow Suppression Risk Prediction",
        "kicker": "Bone Marrow Suppression Risk Prediction System",
        "title": "Hematologic Toxicity Risk by Treatment Cycle",
        "subtitle": "Upload one treatment cycle or use a demo profile to view three Grade ≥3 toxicity risks and their SHAP explanations.",
        "notice_title": "Research-use notice",
        "notice": "This tool has not undergone clinical impact evaluation and must not guide diagnosis, treatment selection, dose adjustment, or replace clinical judgment.",
        "input_step": "1. Enter a treatment cycle",
        "sample": "Demo profile",
        "upload": "Upload CSV",
        "sample_label": "Choose a demo profile",
        "neutral": "Neutral synthetic profile",
        "lower_reserve": "Lower hematologic reserve profile",
        "sample_note": "Demo profiles are identity-free synthetic data and do not represent a typical clinical scenario.",
        "template": "Download demo CSV",
        "upload_label": "Upload a single-row CSV",
        "upload_help": "Exactly one treatment cycle is accepted. Identity, date, and unsupported columns are rejected.",
        "analyze": "Run analysis",
        "running": "Calculating three risks and SHAP explanations…",
        "input_error": "The uploaded input did not pass validation.",
        "prediction_error": "Prediction or SHAP validation failed. No result was returned.",
        "results_step": "2. Risk prediction results",
        "threshold": "Threshold",
        "high": "Above alert threshold",
        "low": "Below alert threshold",
        "explain_step": "3. SHAP risk explanation",
        "endpoint_select": "Outcome to explain",
        "explain_note": "Red increases predicted risk and blue decreases it. SHAP values are predictive contributions, not causal treatment effects.",
        "waterfall_tab": "SHAP waterfall",
        "overview_tab": "Contribution overview",
        "waterfall_title": "Cumulative contributions from model baseline to this prediction",
        "overview_title": "Main feature contribution directions",
        "waterfall_xaxis": "Locked calibrated logit",
        "contribution_xaxis": "Locked calibrated logit contribution",
        "other": "Other features",
        "output": "Prediction",
        "report_step": "4. Download report",
        "report_note": "The report contains only the three risk results and SHAP contributions. It excludes names, IDs, dates, and the complete input row.",
        "report_button": "Download English PDF report",
        "footer": "Research-use prototype · Risk alerts are not clinical recommendations",
        "asset_error": "Model resource validation failed. Prediction is disabled.",
    },
}


st.set_page_config(
    page_title="Bone Marrow Suppression Risk Prediction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .stApp { background: #0E1117; }
    .block-container { max-width: 1160px; padding-top: 2.2rem; padding-bottom: 4rem; }
    .hero-kicker { color: #AAB4C3; font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .hero-title { color: #FAFAFA; font-size: clamp(2.15rem, 4vw, 3.25rem); font-weight: 720; line-height: 1.12; margin: .35rem 0 .65rem; }
    .hero-subtitle { color: #C5CCD6; font-size: 1rem; line-height: 1.65; max-width: 800px; }
    .research-banner { border-left: 4px solid #FF4B4B; background: #171C24; padding: .85rem 1rem; border-radius: 6px; margin: 1.25rem 0 1.8rem; color: #D8DEE8; }
    .section-title { color: #FAFAFA; font-size: 1.15rem; font-weight: 700; margin: 1.55rem 0 .65rem; }
    .risk-card { background: #171C24; border: 1px solid #303742; border-radius: 8px; padding: 1rem 1.05rem; min-height: 180px; }
    .risk-title { color: #C5CCD6; font-size: .86rem; line-height: 1.4; min-height: 2.7rem; }
    .risk-prob { color: #FAFAFA; font-size: 2.15rem; font-weight: 720; line-height: 1.1; margin: .5rem 0 .25rem; }
    .risk-threshold { color: #8F99A8; font-size: .78rem; }
    .status-alert, .status-clear { display: inline-block; margin-top: .75rem; padding: .3rem .55rem; border-radius: 4px; font-size: .78rem; font-weight: 700; }
    .status-alert { background: rgba(255,75,75,.16); color: #FF8585; }
    .status-clear { background: rgba(33,195,128,.14); color: #78D7AD; }
    .report-note { color: #AAB4C3; font-size: .84rem; line-height: 1.55; }
    .footer-note { color: #7F8997; font-size: .78rem; margin-top: 2rem; }
    div[data-testid="stFileUploaderDropzone"] { border-radius: 8px; background: #171C24; border-color: #3A424E; }
    div[data-testid="stButton"] button[kind="primary"], div[data-testid="stDownloadButton"] button[kind="primary"] { font-weight: 700; }
    @media (max-width: 700px) {
        .block-container { padding: 3.6rem 1rem 3rem; }
        .hero-title { font-size: 2rem; }
        .hero-subtitle { font-size: .94rem; }
        .research-banner { margin-bottom: 1.25rem; }
        .risk-card { min-height: auto; margin-bottom: .65rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_predictor() -> FrozenWebPredictor:
    return FrozenWebPredictor(WEB_ROOT / ".artifacts", WEB_ROOT / "artifact_contract.json")


def _validation_message(error: InputValidationError, language: str) -> str:
    if language == "zh":
        return str(error)
    message = str(error)
    if "身份" in message:
        return "Identity-related columns are not allowed."
    if "日期" in message:
        return "Date or time columns are not allowed."
    if "1 个治疗周期" in message:
        return "Upload exactly one treatment-cycle row."
    if "非冻结" in message:
        return "The file contains unsupported columns."
    if "2 MB" in message:
        return "The CSV exceeds the 2 MB limit."
    return TEXT["en"]["input_error"]


def _ordered_contributions(prediction, count: int = 10) -> tuple[list[tuple[str, float]], float]:
    ordered = sorted(
        prediction.raw_feature_shap.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return ordered[:count], sum(value for _, value in ordered[count:])


def contribution_figure(prediction, language: str) -> go.Figure:
    selected, _ = _ordered_contributions(prediction, count=10)
    selected = list(reversed(selected))
    values = [value for _, value in selected]
    label_cutoff = sorted((abs(value) for value in values), reverse=True)[2]
    figure = go.Figure(
        go.Bar(
            orientation="h",
            y=[name for name, _ in selected],
            x=values,
            marker_color=["#FF4B4B" if value >= 0 else "#1F9BFF" for value in values],
            text=[f"{value:+.3f}" if abs(value) >= label_cutoff else "" for value in values],
            textposition="inside",
            insidetextanchor="middle",
            insidetextfont={"color": "#FFFFFF", "size": 10},
            hovertemplate="%{y}<br>%{x:+.6f}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_width=1, line_color="#697386")
    figure.update_layout(
        title={"text": TEXT[language]["overview_title"], "font": {"size": 16}},
        height=430,
        margin={"l": 20, "r": 40, "t": 52, "b": 45},
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        showlegend=False,
        xaxis_title=TEXT[language]["contribution_xaxis"],
        font={"family": "Arial, sans-serif", "size": 12, "color": "#D8DEE8"},
        xaxis={"gridcolor": "#2B323C", "zeroline": False},
        yaxis={"gridcolor": "#2B323C"},
    )
    return figure


def waterfall_figure(prediction, language: str) -> go.Figure:
    selected, other = _ordered_contributions(prediction, count=8)
    labels = [name for name, _ in selected]
    values = [value for _, value in selected]
    if abs(other) > 1e-12:
        labels.append(TEXT[language]["other"])
        values.append(other)
    labels.append(TEXT[language]["output"])
    values.append(prediction.locked_logit)
    measures = ["relative"] * (len(values) - 1) + ["total"]
    visible_text = [
        f"{value:+.3f}" if index < 2 else ""
        for index, value in enumerate(values[:-1])
    ]
    visible_text.append(f"{prediction.locked_logit:.3f}")
    figure = go.Figure(
        go.Waterfall(
            orientation="h",
            base=prediction.locked_logit_base,
            measure=measures,
            y=labels,
            x=values,
            connector={"line": {"color": "#697386", "width": 1}},
            increasing={"marker": {"color": "#FF4B4B"}},
            decreasing={"marker": {"color": "#1F9BFF"}},
            totals={"marker": {"color": "#8B7CF6"}},
            text=visible_text,
            textposition="inside",
            insidetextanchor="middle",
            insidetextfont={"color": "#FFFFFF", "size": 10},
            hovertemplate="%{y}<br>%{x:+.6f}<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": TEXT[language]["waterfall_title"], "font": {"size": 16}},
        height=470,
        margin={"l": 20, "r": 40, "t": 52, "b": 45},
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        showlegend=False,
        xaxis_title=TEXT[language]["waterfall_xaxis"],
        font={"family": "Arial, sans-serif", "size": 12, "color": "#D8DEE8"},
        xaxis={"gridcolor": "#2B323C", "zeroline": False},
        yaxis={"gridcolor": "#2B323C", "autorange": "reversed"},
    )
    return figure


if "language" not in st.session_state:
    st.session_state["language"] = "zh"
language = st.segmented_control(
    "Language",
    options=("zh", "en"),
    default=st.session_state["language"],
    format_func=lambda value: "中文" if value == "zh" else "English",
    selection_mode="single",
    label_visibility="collapsed",
    width="content",
)
language = language or st.session_state["language"]
st.session_state["language"] = language
text = TEXT[language]

st.markdown(f'<div class="hero-kicker">{text["kicker"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-title">{text["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-subtitle">{text["subtitle"]}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="research-banner"><strong>{text["notice_title"]}：</strong> {text["notice"]}</div>',
    unsafe_allow_html=True,
)

try:
    predictor = load_predictor()
except ArtifactValidationError as error:
    st.error(text["asset_error"])
    st.caption(str(error) if language == "zh" else "Please contact the research team.")
    st.stop()
except Exception:
    LOGGER.exception("predictor initialization failed")
    st.error(text["asset_error"])
    st.stop()

st.markdown(f'<div class="section-title">{text["input_step"]}</div>', unsafe_allow_html=True)
source = st.radio(
    "source",
    options=("sample", "upload"),
    format_func=lambda value: text[value],
    horizontal=True,
    label_visibility="collapsed",
)

frame = None
source_name = None
if source == "sample":
    profile = st.selectbox(
        text["sample_label"],
        options=("neutral", "lower_reserve"),
        format_func=lambda value: text[value],
    )
    frame = synthetic_frame(predictor.raw_feature_order, profile=profile)
    source_name = f"synthetic_{profile}"
    st.caption(text["sample_note"])
    st.download_button(
        text["template"],
        frame_to_csv(frame),
        file_name=f"synthetic_{profile}_canonical_input.csv",
        mime="text/csv",
    )
else:
    uploaded = st.file_uploader(
        text["upload_label"],
        type=["csv"],
        help=text["upload_help"],
    )
    if uploaded is not None:
        try:
            frame, _ = prepare_uploaded_frame(uploaded.getvalue(), predictor.raw_feature_order)
            source_name = "uploaded_csv"
        except InputValidationError as error:
            st.error(_validation_message(error, language))

if frame is None:
    st.session_state.pop("predictions", None)
    st.session_state.pop("prediction_source", None)
else:
    input_signature = hashlib.sha256(frame_to_csv(frame)).hexdigest()
    if st.session_state.get("input_signature") != input_signature:
        st.session_state["input_signature"] = input_signature
        st.session_state.pop("predictions", None)
        st.session_state.pop("prediction_source", None)

analyze = st.button(
    text["analyze"],
    type="primary",
    disabled=frame is None,
    width="stretch",
)
if analyze and frame is not None:
    try:
        with st.spinner(text["running"]):
            st.session_state["predictions"] = predictor.predict_all(frame)
            st.session_state["prediction_source"] = source_name
    except (ValueError, InputValidationError) as error:
        st.error(_validation_message(InputValidationError(str(error)), language))
    except Exception:
        LOGGER.exception("prediction failed")
        st.error(text["prediction_error"])

predictions = st.session_state.get("predictions")
if predictions:
    st.markdown(f'<div class="section-title">{text["results_step"]}</div>', unsafe_allow_html=True)
    card_columns = st.columns(3)
    for column, endpoint in zip(card_columns, ENDPOINTS, strict=True):
        result = predictions[endpoint]
        meta = ENDPOINT_META[endpoint]
        status_class = "status-alert" if result.alert else "status-clear"
        status_text = text["high"] if result.alert else text["low"]
        column.markdown(
            f'<div class="risk-card">'
            f'<div class="risk-title"><strong>{meta["label"]}</strong><br>{meta[language]}</div>'
            f'<div class="risk-prob">{result.locked_probability:.1%}</div>'
            f'<div class="risk-threshold">{text["threshold"]} {result.threshold:.2%}</div>'
            f'<span class="{status_class}">{status_text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f'<div class="section-title">{text["explain_step"]}</div>', unsafe_allow_html=True)
    selected_endpoint = st.selectbox(
        text["endpoint_select"],
        options=ENDPOINTS,
        format_func=lambda value: ENDPOINT_META[value]["label"],
    )
    st.caption(text["explain_note"])
    waterfall_tab, overview_tab = st.tabs([text["waterfall_tab"], text["overview_tab"]])
    with waterfall_tab:
        st.plotly_chart(
            waterfall_figure(predictions[selected_endpoint], language),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
    with overview_tab:
        st.plotly_chart(
            contribution_figure(predictions[selected_endpoint], language),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )

    st.markdown(f'<div class="section-title">{text["report_step"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="report-note">{text["report_note"]}</p>', unsafe_allow_html=True)
    try:
        report_bytes = build_pdf_report(predictions, language=language)
        st.download_button(
            text["report_button"],
            report_bytes,
            file_name=f"bone_marrow_suppression_risk_report_{language}.pdf",
            mime="application/pdf",
            type="primary",
            width="stretch",
        )
    except Exception:
        LOGGER.exception("PDF report generation failed")
        st.error("PDF 报告生成失败。" if language == "zh" else "PDF report generation failed.")

    st.markdown(f'<p class="footer-note">{text["footer"]}</p>', unsafe_allow_html=True)
