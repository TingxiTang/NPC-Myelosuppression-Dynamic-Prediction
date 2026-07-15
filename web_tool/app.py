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
    RegimenValidationError,
    apply_regimen,
    load_regimen_catalog,
    prepare_uploaded_frame,
    selected_drug_ids,
    synthetic_frame,
)
from web_runtime.input_data import frame_to_csv


LOGGER = logging.getLogger(__name__)
WEB_ROOT = Path(__file__).resolve().parent
ENDPOINTS = ("hb", "plt", "wbc_neut")
ENDPOINT_META = {
    "hb": {"label": "Hb", "name": "≥3级贫血"},
    "plt": {"label": "PLT", "name": "≥3级血小板减少"},
    "wbc_neut": {"label": "WBC/Neut", "name": "≥3级白细胞/中性粒细胞减少"},
}
TEXT = {
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
    "upload_help": "仅支持单个治疗周期的一行数据。",
    "regimen_step": "2. 选择本次治疗方案",
    "regimen_label": "本次治疗项目",
    "regimen_help": "可搜索并多选；放射治疗沿用原 Web 的 RT 编码方式。",
    "regimen_empty": "请至少选择一个本次治疗项目后再计算。",
    "scenario_title": "方案情景重算",
    "scenario_note": "每次只显示当前所选方案的一次预测。更改方案并重新计算，可观察模型预测值是否变化；这不是方案优劣比较、换药获益估计或治疗推荐。",
    "regimen_summary": "当前模型输入方案",
    "analyze": "计算当前方案风险",
    "running": "正在计算三项风险与 SHAP 解释…",
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
    "footer": "研究用途原型 · 风险警报不是临床建议",
    "asset_error": "模型或治疗方案资源校验失败，已禁止预测。",
}
FEATURE_LABELS = {
    "age": "年龄",
    "gender": "性别",
    "is_smoking": "吸烟",
    "is_drinking": "饮酒",
    "abo_blood_type": "ABO 血型",
    "c_t_stage": "临床 T 分期",
    "c_n_stage": "临床 N 分期",
    "c_m_stage": "临床 M 分期",
    "clinic_stage": "临床总分期",
    "is_chemo": "本次含化疗",
    "is_target": "本次含靶向治疗",
    "is_immuno": "本次含免疫治疗",
    "is_rt": "本次含放射治疗",
    "cum_chemo": "既往累计化疗次数",
    "cum_target": "既往累计靶向治疗次数",
    "cum_immuno": "既往累计免疫治疗次数",
    "cum_rt": "既往累计放射治疗次数",
    "is_first_cycle": "是否首个治疗周期",
    "drug_id": "本次药物组合",
    "category_id": "本次药物类别组合",
}


st.set_page_config(
    page_title=TEXT["page_title"],
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
    .footer-note { color: #7F8997; font-size: .78rem; margin-top: 2rem; }
    div[data-testid="stFileUploaderDropzone"] { border-radius: 8px; background: #171C24; border-color: #3A424E; }
    div[data-testid="stButton"] button[kind="primary"] { font-weight: 700; }
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


def feature_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("base_"):
        return "治疗前 " + name.removeprefix("base_")
    if name.startswith("prev_nadir_"):
        return "既往最低 " + name.removeprefix("prev_nadir_")
    return name


def _ordered_contributions(prediction, count: int = 10) -> tuple[list[tuple[str, float]], float]:
    ordered = sorted(
        prediction.raw_feature_shap.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return ordered[:count], sum(value for _, value in ordered[count:])


def contribution_figure(prediction) -> go.Figure:
    selected, _ = _ordered_contributions(prediction, count=10)
    selected = list(reversed(selected))
    values = [value for _, value in selected]
    label_cutoff = sorted((abs(value) for value in values), reverse=True)[2]
    figure = go.Figure(
        go.Bar(
            orientation="h",
            y=[feature_label(name) for name, _ in selected],
            x=values,
            marker_color=["#FF4B4B" if value >= 0 else "#1F9BFF" for value in values],
            text=[f"{value:+.3f}" if abs(value) >= label_cutoff else "" for value in values],
            textposition="inside",
            insidetextanchor="middle",
            insidetextfont={"color": "#FFFFFF", "size": 10},
            hovertemplate="%{y}<br>贡献：%{x:+.6f}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_width=1, line_color="#697386")
    figure.update_layout(
        title={"text": TEXT["overview_title"], "font": {"size": 16}},
        height=430,
        margin={"l": 20, "r": 40, "t": 52, "b": 45},
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        showlegend=False,
        xaxis_title=TEXT["contribution_xaxis"],
        font={"family": "Arial, sans-serif", "size": 12, "color": "#D8DEE8"},
        xaxis={"gridcolor": "#2B323C", "zeroline": False},
        yaxis={"gridcolor": "#2B323C"},
    )
    return figure


def waterfall_figure(prediction) -> go.Figure:
    selected, other = _ordered_contributions(prediction, count=8)
    labels = [feature_label(name) for name, _ in selected]
    values = [value for _, value in selected]
    if abs(other) > 1e-12:
        labels.append(TEXT["other"])
        values.append(other)
    labels.append(TEXT["output"])
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
            hovertemplate="%{y}<br>贡献：%{x:+.6f}<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": TEXT["waterfall_title"], "font": {"size": 16}},
        height=470,
        margin={"l": 20, "r": 40, "t": 52, "b": 45},
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        showlegend=False,
        xaxis_title=TEXT["waterfall_xaxis"],
        font={"family": "Arial, sans-serif", "size": 12, "color": "#D8DEE8"},
        xaxis={"gridcolor": "#2B323C", "zeroline": False},
        yaxis={"gridcolor": "#2B323C", "autorange": "reversed"},
    )
    return figure


st.markdown(f'<div class="hero-kicker">{TEXT["kicker"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-title">{TEXT["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-subtitle">{TEXT["subtitle"]}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="research-banner"><strong>{TEXT["notice_title"]}：</strong> {TEXT["notice"]}</div>',
    unsafe_allow_html=True,
)

try:
    predictor = load_predictor()
    regimen_catalog = load_regimen_catalog(predictor.drug_vocab, predictor.category_vocab)
except (ArtifactValidationError, RegimenValidationError) as error:
    st.error(TEXT["asset_error"])
    st.caption(str(error))
    st.stop()
except Exception:
    LOGGER.exception("predictor initialization failed")
    st.error(TEXT["asset_error"])
    st.stop()

regimen_labels = {entry.drug_id: entry.display_name for entry in regimen_catalog}

st.markdown(f'<div class="section-title">{TEXT["input_step"]}</div>', unsafe_allow_html=True)
source = st.radio(
    "数据来源",
    options=("sample", "upload"),
    format_func=lambda value: TEXT[value],
    horizontal=True,
    label_visibility="collapsed",
)

base_frame = None
source_name = None
profile = None
if source == "sample":
    profile = st.selectbox(
        TEXT["sample_label"],
        options=("neutral", "lower_reserve"),
        format_func=lambda value: TEXT[value],
    )
    base_frame = synthetic_frame(predictor.raw_feature_order, profile=profile)
    source_name = f"synthetic_{profile}"
    st.caption(TEXT["sample_note"])
else:
    uploaded = st.file_uploader(
        TEXT["upload_label"],
        type=["csv"],
        help=TEXT["upload_help"],
    )
    if uploaded is not None:
        try:
            base_frame, _ = prepare_uploaded_frame(uploaded.getvalue(), predictor.raw_feature_order)
            source_name = "uploaded_csv"
        except InputValidationError as error:
            st.error(str(error))

frame = None
regimen_selection = None
if base_frame is not None:
    st.markdown(f'<div class="section-title">{TEXT["regimen_step"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="scenario-note"><strong>{TEXT["scenario_title"]}：</strong> {TEXT["scenario_note"]}</div>',
        unsafe_allow_html=True,
    )
    try:
        default_ids = selected_drug_ids(base_frame, regimen_catalog)
        base_signature = hashlib.sha256(frame_to_csv(base_frame)).hexdigest()
        chosen_ids = st.multiselect(
            TEXT["regimen_label"],
            options=tuple(regimen_labels),
            default=default_ids,
            format_func=regimen_labels.__getitem__,
            help=TEXT["regimen_help"],
            key=f"regimen_{base_signature[:16]}",
        )
        if chosen_ids:
            frame, regimen_selection = apply_regimen(base_frame, chosen_ids, regimen_catalog)
            summary = " + ".join(regimen_selection.display_names)
            st.markdown(
                f'<div class="regimen-summary"><strong>{TEXT["regimen_summary"]}：</strong>{summary}</div>',
                unsafe_allow_html=True,
            )
            if source == "sample":
                st.download_button(
                    TEXT["template"],
                    frame_to_csv(frame),
                    file_name=f"synthetic_{profile}_current_regimen.csv",
                    mime="text/csv",
                )
        else:
            st.info(TEXT["regimen_empty"])
    except RegimenValidationError as error:
        st.error(str(error))

if frame is None:
    st.session_state.pop("predictions", None)
    st.session_state.pop("prediction_source", None)
    st.session_state.pop("prediction_regimen", None)
else:
    input_signature = hashlib.sha256(frame_to_csv(frame)).hexdigest()
    if st.session_state.get("input_signature") != input_signature:
        st.session_state["input_signature"] = input_signature
        st.session_state.pop("predictions", None)
        st.session_state.pop("prediction_source", None)
        st.session_state.pop("prediction_regimen", None)

analyze = st.button(
    TEXT["analyze"],
    type="primary",
    disabled=frame is None,
    width="stretch",
)
if analyze and frame is not None and regimen_selection is not None:
    try:
        with st.spinner(TEXT["running"]):
            st.session_state["predictions"] = predictor.predict_all(frame)
            st.session_state["prediction_source"] = source_name
            st.session_state["prediction_regimen"] = regimen_selection.display_names
    except (ValueError, InputValidationError, RegimenValidationError) as error:
        st.error(str(error))
    except Exception:
        LOGGER.exception("prediction failed")
        st.error(TEXT["prediction_error"])

predictions = st.session_state.get("predictions")
if predictions:
    st.markdown(f'<div class="section-title">{TEXT["results_step"]}</div>', unsafe_allow_html=True)
    predicted_regimen = " + ".join(st.session_state.get("prediction_regimen", ()))
    st.markdown(
        f'<div class="regimen-summary"><strong>{TEXT["regimen_summary"]}：</strong>{predicted_regimen}</div>',
        unsafe_allow_html=True,
    )
    card_columns = st.columns(3)
    for column, endpoint in zip(card_columns, ENDPOINTS, strict=True):
        result = predictions[endpoint]
        meta = ENDPOINT_META[endpoint]
        status_class = "status-alert" if result.alert else "status-clear"
        status_text = TEXT["high"] if result.alert else TEXT["low"]
        column.markdown(
            f'<div class="risk-card">'
            f'<div class="risk-title"><strong>{meta["label"]}</strong><br>{meta["name"]}</div>'
            f'<div class="risk-prob">{result.locked_probability:.1%}</div>'
            f'<div class="risk-threshold">{TEXT["threshold"]} {result.threshold:.2%}</div>'
            f'<span class="{status_class}">{status_text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f'<div class="section-title">{TEXT["explain_step"]}</div>', unsafe_allow_html=True)
    selected_endpoint = st.selectbox(
        TEXT["endpoint_select"],
        options=ENDPOINTS,
        format_func=lambda value: ENDPOINT_META[value]["label"],
    )
    st.caption(TEXT["explain_note"])
    waterfall_tab, overview_tab = st.tabs([TEXT["waterfall_tab"], TEXT["overview_tab"]])
    with waterfall_tab:
        st.plotly_chart(
            waterfall_figure(predictions[selected_endpoint]),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
    with overview_tab:
        st.plotly_chart(
            contribution_figure(predictions[selected_endpoint]),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )

    st.markdown(f'<p class="footer-note">{TEXT["footer"]}</p>', unsafe_allow_html=True)
