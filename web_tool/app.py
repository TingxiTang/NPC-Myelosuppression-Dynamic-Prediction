"""Streamlit research-use interface for the frozen three-endpoint models."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from web_runtime.input_data import (
    InputValidationError,
    frame_to_csv,
    prepare_uploaded_frame,
    synthetic_frame,
)
from web_runtime.predictor import ArtifactValidationError, FrozenWebPredictor
from web_runtime.regimen import (
    RegimenValidationError,
    apply_regimen,
    load_regimen_catalog,
    selected_drug_ids,
)


LOGGER = logging.getLogger(__name__)
WEB_ROOT = Path(__file__).resolve().parent
ENDPOINTS = ("hb", "plt", "wbc_neut")
ENDPOINT_META = {
    "hb": {"label": "Hb", "zh": "≥3级贫血", "en": "Grade ≥3 anemia"},
    "plt": {"label": "PLT", "zh": "≥3级血小板减少", "en": "Grade ≥3 thrombocytopenia"},
    "wbc_neut": {
        "label": "WBC/Neut",
        "zh": "≥3级白细胞/中性粒细胞减少",
        "en": "Grade ≥3 leukopenia/neutropenia",
    },
}
TEXT = {
    "zh": {
        "page_title": "骨髓抑制风险预测系统",
        "kicker": "骨髓抑制风险预测系统",
        "title": "治疗周期血液学毒性风险预测",
        "subtitle": "输入一个治疗周期和本次治疗方案，查看三项 ≥3级血液学毒性风险及 SHAP 解释。",
        "notice_title": "研究用途提示",
        "notice": "本工具未经临床影响评估，不用于诊断、治疗选择、剂量调整或替代临床判断。",
        "input_step": "1. 输入治疗周期",
        "sample": "演示样例",
        "upload": "上传 CSV",
        "sample_label": "选择演示样例",
        "neutral": "中性合成样例",
        "lower_reserve": "较低血液学储备样例",
        "sample_note": "演示样例为无身份合成数据，不代表典型临床场景。",
        "template": "下载当前合成样例 CSV",
        "upload_label": "上传单行 CSV",
        "upload_help": "仅支持单个治疗周期的一行数据；身份、日期及非模型字段会被拒绝。",
        "regimen_step": "2. 选择本次治疗方案",
        "regimen_label": "本次治疗项目",
        "regimen_help": "可搜索并多选；放射治疗沿用原 Web 的 RT 编码方式。",
        "regimen_empty": "请至少选择一个本次治疗项目后再计算。",
        "scenario_title": "方案情景重算",
        "scenario_note": "每次只显示当前所选方案的一次预测。更改方案并重新计算，可观察模型预测值是否变化；这不是方案优劣比较、换药获益估计或治疗推荐。",
        "regimen_summary": "当前模型输入方案",
        "analyze": "计算当前方案风险",
        "running": "正在计算三项风险与 SHAP 解释…",
        "input_error": "输入未通过校验。",
        "regimen_error": "治疗方案未通过校验。",
        "prediction_error": "预测或 SHAP 校验失败，已停止输出。",
        "results_step": "3. 风险预测结果",
        "threshold": "阈值",
        "high": "达到警报阈值",
        "low": "未达到警报阈值",
        "explain_step": "4. SHAP 风险解释",
        "endpoint_select": "选择要解释的结局",
        "explain_note": "红色提高预测风险，蓝色降低预测风险；SHAP 表示预测贡献，不代表治疗因果效应。",
        "waterfall_tab": "SHAP 瀑布图",
        "overview_tab": "贡献概览",
        "waterfall_title": "从模型基线到本次预测的累计贡献",
        "overview_title": "主要特征贡献方向",
        "waterfall_xaxis": "锁定校准对数优势",
        "contribution_xaxis": "锁定校准对数优势贡献",
        "other": "其余特征",
        "output": "本次预测",
        "report_step": "5. 下载报告",
        "report_note": "PDF 仅包含三项风险结果和 SHAP 贡献，不包含姓名、ID、日期、治疗方案或完整输入行。",
        "report_button": "下载中文 PDF 报告",
        "pdf_error": "PDF 报告生成失败。",
        "footer": "研究用途原型 · 风险警报不是临床建议",
        "asset_error": "模型或治疗方案资源校验失败，已禁止预测。",
    },
    "en": {
        "page_title": "Bone Marrow Suppression Risk Prediction",
        "kicker": "Bone Marrow Suppression Risk Prediction System",
        "title": "Hematologic Toxicity Risk by Treatment Cycle",
        "subtitle": "Enter one treatment cycle and its current regimen to view three Grade ≥3 hematologic toxicity risks and SHAP explanations.",
        "notice_title": "Research-use notice",
        "notice": "This tool has not undergone clinical impact evaluation and must not guide diagnosis, treatment selection, dose adjustment, or replace clinical judgment.",
        "input_step": "1. Enter a treatment cycle",
        "sample": "Demo profile",
        "upload": "Upload CSV",
        "sample_label": "Choose a demo profile",
        "neutral": "Neutral synthetic profile",
        "lower_reserve": "Lower hematologic reserve profile",
        "sample_note": "Demo profiles are identity-free synthetic data and do not represent a typical clinical scenario.",
        "template": "Download the current synthetic CSV",
        "upload_label": "Upload a single-row CSV",
        "upload_help": "Exactly one treatment-cycle row is accepted. Identity, date, and unsupported columns are rejected.",
        "regimen_step": "2. Select the current regimen",
        "regimen_label": "Current treatment items",
        "regimen_help": "Search and select one or more items. Radiotherapy retains the audited legacy RT encoding.",
        "regimen_empty": "Select at least one current treatment item before calculation.",
        "scenario_title": "Regimen scenario recalculation",
        "scenario_note": "Only one prediction for the currently selected regimen is shown. Change the regimen and recalculate to observe whether model predictions change. This is not a regimen comparison, treatment-benefit estimate, or treatment recommendation.",
        "regimen_summary": "Current model-input regimen",
        "analyze": "Calculate risk for this regimen",
        "running": "Calculating three risks and SHAP explanations…",
        "input_error": "The input did not pass validation.",
        "regimen_error": "The treatment regimen did not pass validation.",
        "prediction_error": "Prediction or SHAP validation failed. No result was returned.",
        "results_step": "3. Risk prediction results",
        "threshold": "Threshold",
        "high": "Above alert threshold",
        "low": "Below alert threshold",
        "explain_step": "4. SHAP risk explanation",
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
        "report_step": "5. Download report",
        "report_note": "The PDF contains only the three risk results and SHAP contributions. It excludes names, IDs, dates, the treatment regimen, and the complete input row.",
        "report_button": "Download English PDF report",
        "pdf_error": "PDF report generation failed.",
        "footer": "Research-use prototype · Risk alerts are not clinical recommendations",
        "asset_error": "Model or regimen resources failed validation. Prediction is disabled.",
    },
}
FEATURE_LABELS = {
    "zh": {
        "age": "年龄", "gender": "性别", "is_smoking": "吸烟", "is_drinking": "饮酒",
        "abo_blood_type": "ABO 血型", "c_t_stage": "临床 T 分期", "c_n_stage": "临床 N 分期",
        "c_m_stage": "临床 M 分期", "clinic_stage": "临床总分期", "is_chemo": "本次含化疗",
        "is_target": "本次含靶向治疗", "is_immuno": "本次含免疫治疗", "is_rt": "本次含放射治疗",
        "cum_chemo": "既往累计化疗次数", "cum_target": "既往累计靶向治疗次数",
        "cum_immuno": "既往累计免疫治疗次数", "cum_rt": "既往累计放射治疗次数",
        "is_first_cycle": "是否首个治疗周期", "drug_id": "本次药物组合", "category_id": "本次药物类别组合",
    },
    "en": {
        "age": "Age", "gender": "Sex", "is_smoking": "Smoking history", "is_drinking": "Alcohol history",
        "abo_blood_type": "ABO blood group", "c_t_stage": "Clinical T stage", "c_n_stage": "Clinical N stage",
        "c_m_stage": "Clinical M stage", "clinic_stage": "Overall clinical stage", "is_chemo": "Current chemotherapy",
        "is_target": "Current targeted therapy", "is_immuno": "Current immunotherapy", "is_rt": "Current radiotherapy",
        "cum_chemo": "Prior chemotherapy count", "cum_target": "Prior targeted-therapy count",
        "cum_immuno": "Prior immunotherapy count", "cum_rt": "Prior radiotherapy count",
        "is_first_cycle": "First treatment cycle", "drug_id": "Current drug combination", "category_id": "Current drug categories",
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
    .hero-kicker { color: #AAB4C3; font-size: .78rem; font-weight: 700; letter-spacing: .08em; }
    .hero-title { color: #FAFAFA; font-size: clamp(2.15rem, 4vw, 3.25rem); font-weight: 720; line-height: 1.12; margin: .35rem 0 .65rem; }
    .hero-subtitle { color: #C5CCD6; font-size: 1rem; line-height: 1.65; max-width: 820px; }
    .research-banner { border-left: 4px solid #FF4B4B; background: #171C24; padding: .85rem 1rem; border-radius: 6px; margin: 1.25rem 0 1.8rem; color: #D8DEE8; }
    .scenario-note { border: 1px solid #303742; background: #171C24; padding: .8rem .95rem; border-radius: 7px; margin: .45rem 0 .85rem; color: #C5CCD6; font-size: .88rem; line-height: 1.55; }
    .section-title { color: #FAFAFA; font-size: 1.15rem; font-weight: 700; margin: 1.55rem 0 .65rem; }
    .regimen-summary { color: #D8DEE8; font-size: .9rem; margin: .35rem 0 1rem; }
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


def _validation_message(error: Exception, language: str) -> str:
    if language == "zh":
        return str(error)
    message = str(error)
    if "身份" in message:
        return "Identity-related columns are not allowed."
    if "日期" in message:
        return "Date or time columns are not allowed."
    if "1 个治疗周期" in message or "仅含一行" in message:
        return "Provide exactly one treatment-cycle row."
    if "非冻结" in message:
        return "The file contains unsupported columns."
    if "2 MB" in message:
        return "The CSV exceeds the 2 MB limit."
    if "至少选择" in message:
        return TEXT["en"]["regimen_empty"]
    if "治疗方案字典" in message or "冻结治疗编码" in message:
        return "The audited treatment-regimen dictionary failed validation."
    if "用药" in message or "方案字段" in message:
        return TEXT["en"]["regimen_error"]
    return TEXT["en"]["input_error"]


def feature_label(name: str, language: str) -> str:
    if name in FEATURE_LABELS[language]:
        return FEATURE_LABELS[language][name]
    if name.startswith("base_"):
        prefix = "治疗前 " if language == "zh" else "Pre-treatment "
        return prefix + name.removeprefix("base_")
    if name.startswith("prev_nadir_"):
        prefix = "既往最低 " if language == "zh" else "Prior nadir "
        return prefix + name.removeprefix("prev_nadir_")
    return name


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
            y=[feature_label(name, language) for name, _ in selected],
            x=values,
            marker_color=["#FF4B4B" if value >= 0 else "#1F9BFF" for value in values],
            text=[f"{value:+.3f}" if abs(value) >= label_cutoff else "" for value in values],
            textposition="inside",
            insidetextanchor="middle",
            insidetextfont={"color": "#FFFFFF", "size": 10},
            hovertemplate=(
                "%{y}<br>贡献：%{x:+.6f}<extra></extra>"
                if language == "zh"
                else "%{y}<br>Contribution: %{x:+.6f}<extra></extra>"
            ),
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
    labels = [feature_label(name, language) for name, _ in selected]
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
            hovertemplate=(
                "%{y}<br>贡献：%{x:+.6f}<extra></extra>"
                if language == "zh"
                else "%{y}<br>Contribution: %{x:+.6f}<extra></extra>"
            ),
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
separator = "：" if language == "zh" else ": "
if st.session_state.get("ui_contract_version") != "bilingual_pdf_regimen_v1":
    st.session_state["ui_contract_version"] = "bilingual_pdf_regimen_v1"
    st.session_state.pop("predictions", None)
    st.session_state.pop("prediction_source", None)
    st.session_state.pop("prediction_regimen", None)
    st.session_state.pop("prediction_regimen_ids", None)

st.markdown(f'<div class="hero-kicker">{text["kicker"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-title">{text["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-subtitle">{text["subtitle"]}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="research-banner"><strong>{text["notice_title"]}{separator}</strong> {text["notice"]}</div>',
    unsafe_allow_html=True,
)

try:
    predictor = load_predictor()
    regimen_catalog = load_regimen_catalog(predictor.drug_vocab, predictor.category_vocab)
except (ArtifactValidationError, RegimenValidationError) as error:
    st.error(text["asset_error"])
    st.caption(_validation_message(error, language))
    st.stop()
except Exception:
    LOGGER.exception("predictor initialization failed")
    st.error(text["asset_error"])
    st.stop()

regimen_labels = {
    entry.drug_id: entry.display_name_for(language) for entry in regimen_catalog
}

st.markdown(f'<div class="section-title">{text["input_step"]}</div>', unsafe_allow_html=True)
source = st.radio(
    "Data source",
    options=("sample", "upload"),
    format_func=lambda value: text[value],
    horizontal=True,
    label_visibility="collapsed",
)

base_frame = None
source_name = None
profile = None
if source == "sample":
    profile = st.selectbox(
        text["sample_label"],
        options=("neutral", "lower_reserve"),
        format_func=lambda value: text[value],
    )
    base_frame = synthetic_frame(predictor.raw_feature_order, profile=profile)
    source_name = f"synthetic_{profile}"
    st.caption(text["sample_note"])
else:
    uploaded = st.file_uploader(
        text["upload_label"],
        type=["csv"],
        help=text["upload_help"],
    )
    if uploaded is not None:
        try:
            base_frame, _ = prepare_uploaded_frame(uploaded.getvalue(), predictor.raw_feature_order)
            source_name = "uploaded_csv"
        except InputValidationError as error:
            st.error(_validation_message(error, language))

frame = None
regimen_selection = None
if base_frame is not None:
    st.markdown(f'<div class="section-title">{text["regimen_step"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="scenario-note"><strong>{text["scenario_title"]}{separator}</strong> {text["scenario_note"]}</div>',
        unsafe_allow_html=True,
    )
    try:
        default_ids = selected_drug_ids(base_frame, regimen_catalog)
        base_signature = hashlib.sha256(frame_to_csv(base_frame)).hexdigest()
        chosen_ids = st.multiselect(
            text["regimen_label"],
            options=tuple(regimen_labels),
            default=default_ids,
            format_func=regimen_labels.__getitem__,
            help=text["regimen_help"],
            key=f"regimen_{base_signature[:16]}",
        )
        if chosen_ids:
            frame, regimen_selection = apply_regimen(base_frame, chosen_ids, regimen_catalog)
            summary = " + ".join(
                regimen_labels[drug_id] for drug_id in regimen_selection.drug_ids
            )
            st.markdown(
                f'<div class="regimen-summary"><strong>{text["regimen_summary"]}{separator}</strong>{summary}</div>',
                unsafe_allow_html=True,
            )
            if source == "sample":
                st.download_button(
                    text["template"],
                    frame_to_csv(frame),
                    file_name=f"synthetic_{profile}_current_regimen.csv",
                    mime="text/csv",
                )
        else:
            st.info(text["regimen_empty"])
    except RegimenValidationError as error:
        st.error(_validation_message(error, language))

if frame is None:
    st.session_state.pop("predictions", None)
    st.session_state.pop("prediction_source", None)
    st.session_state.pop("prediction_regimen", None)
    st.session_state.pop("prediction_regimen_ids", None)
else:
    input_signature = hashlib.sha256(frame_to_csv(frame)).hexdigest()
    if st.session_state.get("input_signature") != input_signature:
        st.session_state["input_signature"] = input_signature
        st.session_state.pop("predictions", None)
        st.session_state.pop("prediction_source", None)
        st.session_state.pop("prediction_regimen", None)
        st.session_state.pop("prediction_regimen_ids", None)

analyze = st.button(
    text["analyze"],
    type="primary",
    disabled=frame is None,
    width="stretch",
)
if analyze and frame is not None and regimen_selection is not None:
    try:
        with st.spinner(text["running"]):
            st.session_state["predictions"] = predictor.predict_all(frame)
            st.session_state["prediction_source"] = source_name
            st.session_state["prediction_regimen_ids"] = regimen_selection.drug_ids
    except (ValueError, InputValidationError, RegimenValidationError) as error:
        st.error(_validation_message(error, language))
    except Exception:
        LOGGER.exception("prediction failed")
        st.error(text["prediction_error"])

predictions = st.session_state.get("predictions")
if predictions:
    st.markdown(f'<div class="section-title">{text["results_step"]}</div>', unsafe_allow_html=True)
    predicted_regimen = " + ".join(
        regimen_labels[drug_id]
        for drug_id in st.session_state.get("prediction_regimen_ids", ())
    )
    st.markdown(
        f'<div class="regimen-summary"><strong>{text["regimen_summary"]}{separator}</strong>{predicted_regimen}</div>',
        unsafe_allow_html=True,
    )
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
        from web_runtime.reporting import build_pdf_report

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
        st.error(text["pdf_error"])

    st.markdown(f'<p class="footer-note">{text["footer"]}</p>', unsafe_allow_html=True)
