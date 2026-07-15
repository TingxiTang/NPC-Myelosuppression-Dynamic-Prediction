"""Streamlit research-use interface for the frozen three-endpoint models."""

from __future__ import annotations

import logging
import hashlib
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from web_runtime import (
    ArtifactValidationError,
    FrozenWebPredictor,
    InputValidationError,
    prepare_uploaded_frame,
    synthetic_frame,
)
from web_runtime.input_data import frame_to_csv


LOGGER = logging.getLogger(__name__)
WEB_ROOT = Path(__file__).resolve().parent
ENDPOINT_META = {
    "hb": {"label": "Hb", "title": "Grade ≥3 贫血", "color": "#4C78A8"},
    "plt": {"label": "PLT", "title": "Grade ≥3 血小板减少", "color": "#B07AA1"},
    "wbc_neut": {"label": "WBC/Neut", "title": "Grade ≥3 白细胞/中性粒细胞减少", "color": "#F28E2B"},
}


st.set_page_config(
    page_title="急性骨髓抑制研究风险原型",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
    .stApp { background: linear-gradient(180deg, #F7F8FA 0%, #FFFFFF 36%); }
    .block-container { max-width: 1180px; padding-top: 4.25rem; padding-bottom: 3rem; }
    .hero-kicker { color: #2F4B7C; font-size: .78rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; }
    .hero-title { color: #17212B; font-size: clamp(2rem, 5vw, 3.5rem); font-weight: 760; line-height: 1.05; margin: .35rem 0 .75rem; }
    .hero-subtitle { color: #536171; font-size: 1.02rem; line-height: 1.7; max-width: 780px; }
    .research-banner { border: 1px solid #D5DEE8; border-left: 5px solid #2F4B7C; background: #F3F6F9; padding: 1rem 1.1rem; border-radius: 12px; margin: 1.4rem 0; color: #293747; }
    .section-label { color: #2F4B7C; font-weight: 750; font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; margin-bottom: .2rem; }
    .risk-card { background: #FFFFFF; border: 1px solid #E1E6EB; border-radius: 16px; padding: 1.05rem 1.1rem; min-height: 210px; box-shadow: 0 8px 24px rgba(29,45,61,.05); }
    .risk-title { color: #536171; font-size: .86rem; line-height: 1.35; min-height: 2.4rem; }
    .risk-prob { color: #17212B; font-size: 2.35rem; font-weight: 760; line-height: 1.1; margin: .55rem 0 .25rem; }
    .risk-threshold { color: #6A7684; font-size: .8rem; }
    .status-alert { display: inline-block; margin-top: .85rem; padding: .35rem .65rem; border-radius: 999px; background: #FFF0EC; color: #A13D2D; font-size: .8rem; font-weight: 700; }
    .status-clear { display: inline-block; margin-top: .85rem; padding: .35rem .65rem; border-radius: 999px; background: #EAF5EF; color: #246B47; font-size: .8rem; font-weight: 700; }
    .tiny-note { color: #6A7684; font-size: .78rem; line-height: 1.55; }
    div[data-testid="stFileUploaderDropzone"] { border-radius: 14px; border-color: #B8C5D2; background: #FBFCFD; }
    div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E1E6EB; border-radius: 12px; padding: .75rem; }
    @media (max-width: 700px) {
        .block-container { padding-top: 3.75rem; }
        .hero-title { font-size: 2rem; }
        .risk-card { min-height: auto; margin-bottom: .7rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="正在校验冻结模型与预处理资产…")
def load_predictor() -> FrozenWebPredictor:
    return FrozenWebPredictor(
        WEB_ROOT / ".artifacts",
        WEB_ROOT / "artifact_contract.json",
    )


def missing_audit(frame: pd.DataFrame, source: str) -> dict[str, object]:
    missing = [
        column
        for column in frame.columns
        if column not in {"drug_id", "category_id"}
        and pd.isna(frame.iloc[0][column])
    ]
    return {
        "source": source,
        "n_expected_features": len(frame.columns),
        "n_missing_values": len(missing),
        "missing_values": missing,
        "privacy_check": "synthetic_identity_free" if source.startswith("synthetic") else "passed",
    }


def waterfall_figure(prediction, color: str) -> go.Figure:
    ordered = sorted(
        prediction.raw_feature_shap.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    selected = ordered[:10]
    other = sum(value for _, value in ordered[10:])
    labels = ["基线 logit", *[name for name, _ in selected], "其余特征", "锁定 logit"]
    values = [
        prediction.locked_logit_base,
        *[value for _, value in selected],
        other,
        prediction.locked_logit,
    ]
    measures = ["absolute", *(["relative"] * (len(selected) + 1)), "total"]
    figure = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            connector={"line": {"color": "#AAB4BF", "width": 1}},
            increasing={"marker": {"color": color}},
            decreasing={"marker": {"color": "#4F6D7A"}},
            totals={"marker": {"color": "#2F4B7C"}},
            hovertemplate="%{x}<br>logit 贡献: %{y:.5f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=410,
        margin={"l": 20, "r": 20, "t": 28, "b": 90},
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        showlegend=False,
        yaxis_title="锁定校准 logit",
        xaxis={"tickangle": -38, "automargin": True},
        font={"family": "Arial, sans-serif", "size": 12, "color": "#293747"},
    )
    return figure


st.markdown('<div class="hero-kicker">Research-use Web prototype · Frozen XGBoost</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">治疗周期血液学毒性风险原型</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">使用预先锁定的三个 XGBoost endpoint，展示单个治疗周期的研究风险概率、固定阈值状态与局部特征贡献。</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="research-banner"><strong>仅用于研究和技术验证。</strong> '
    '本原型未经前瞻性临床影响评估，不用于诊断、治疗选择、剂量调整或代替临床判断。高警报负担、低阳性预测值和外部来源异质性仍限制实时部署。</div>',
    unsafe_allow_html=True,
)

try:
    predictor = load_predictor()
except ArtifactValidationError as error:
    st.error("冻结资产校验未通过，已禁止预测。")
    st.caption(str(error))
    st.stop()
except Exception:
    LOGGER.exception("predictor initialization failed")
    st.error("模型资源无法安全加载，已禁止预测。")
    st.stop()

summary = predictor.technical_summary()
with st.sidebar:
    st.markdown("### 冻结实现状态")
    st.success("资产哈希校验通过")
    st.caption(f"Bundle: `{summary['bundle_id']}`")
    st.caption(f"Selection lock: `{str(summary['selection_lock_sha256'])[:12]}…`")
    st.caption(f"特征：{summary['raw_features']} raw → {summary['encoded_features']} encoded")
    st.caption("SHAP 空间：锁定校准 logit")
    st.markdown("---")
    st.markdown("### 输入边界")
    st.caption("仅处理单行、单周期 CSV；姓名、ID、日期、地址和自由文本字段会被拒绝。")
    st.caption("上传数据仅在当前 Streamlit 会话内存中处理，程序不写入患者输入。")

st.markdown('<div class="section-label">01 · 输入一个治疗周期</div>', unsafe_allow_html=True)
source_mode = st.radio(
    "数据来源",
    ["合成演示样例", "上传 canonical CSV"],
    horizontal=True,
    label_visibility="collapsed",
)

frame = None
input_audit = None
if source_mode == "合成演示样例":
    profile_label = st.selectbox(
        "选择完全合成的技术测试输入",
        ["中性合成样例", "较低血液学储备合成样例"],
    )
    profile = "neutral" if profile_label == "中性合成样例" else "lower_reserve"
    frame = synthetic_frame(predictor.raw_feature_order, profile=profile)
    input_audit = missing_audit(frame, f"synthetic_{profile}")
    st.info("该样例由固定规则生成，不是患者记录，不代表典型临床场景。")
    st.download_button(
        "下载合成 CSV 模板",
        frame_to_csv(frame),
        file_name=f"synthetic_{profile}_canonical_input.csv",
        mime="text/csv",
    )
else:
    uploaded = st.file_uploader(
        "上传无身份字段的单行 CSV",
        type=["csv"],
        help="缺失的冻结特征列将按 canonical 缺失处理补齐；非冻结列和疑似身份/日期列会被拒绝。",
    )
    if uploaded is not None:
        try:
            frame, input_audit = prepare_uploaded_frame(
                uploaded.getvalue(), predictor.raw_feature_order
            )
        except InputValidationError as error:
            st.error(str(error))

if frame is None:
    st.session_state.pop("predictions", None)
    st.session_state.pop("prediction_source", None)

if frame is not None and input_audit is not None:
    input_signature = hashlib.sha256(frame_to_csv(frame)).hexdigest()
    if st.session_state.get("input_signature") != input_signature:
        st.session_state["input_signature"] = input_signature
        st.session_state.pop("predictions", None)
        st.session_state.pop("prediction_source", None)
    missing_count = int(input_audit["n_missing_values"])
    cols = st.columns(3)
    cols[0].metric("冻结 raw 特征", len(predictor.raw_feature_order))
    cols[1].metric("当前缺失值", missing_count)
    cols[2].metric("隐私门", "通过")
    if missing_count:
        with st.expander(f"查看 {missing_count} 个缺失特征"):
            st.write(", ".join(input_audit["missing_values"]))
            st.caption("仅使用冻结 encoder/preprocessor 中的既定缺失机制；程序不会用 0 或新中位数自行填补。")
    with st.expander("输入摘要（不含身份字段）"):
        preview_fields = [
            "age", "gender", "clinic_stage", "base_Hb", "base_PLT", "base_WBC",
            "base_Neut", "prev_nadir_Hb", "prev_nadir_PLT", "prev_nadir_WBC",
            "drug_id", "category_id",
        ]
        st.dataframe(frame.loc[:, preview_fields], width="stretch", hide_index=True)

st.markdown('<div class="section-label">02 · 锁定三结局计算</div>', unsafe_allow_html=True)
analyze = st.button(
    "运行冻结预测与局部解释",
    type="primary",
    disabled=frame is None,
    width="stretch",
)

if analyze and frame is not None:
    try:
        with st.spinner("正在计算三个 endpoint 并校验 SHAP 加和…"):
            st.session_state["predictions"] = predictor.predict_all(frame)
            st.session_state["prediction_source"] = input_audit["source"]
    except (ValueError, InputValidationError) as error:
        st.error(f"输入未通过 canonical 特征校验：{error}")
    except Exception:
        LOGGER.exception("prediction failed")
        st.error("预测或 SHAP 一致性校验失败，已停止输出。")

predictions = st.session_state.get("predictions")
if predictions:
    st.markdown('<div class="section-label">03 · 三结局锁定风险</div>', unsafe_allow_html=True)
    st.caption("概率均为 tune 阶段固定 logistic recalibrator 的输出；分类使用冻结的 `>=` 阈值，未在当前输入上重校准或调阈值。")
    card_columns = st.columns(3)
    for column, endpoint in zip(card_columns, ("hb", "plt", "wbc_neut"), strict=True):
        result = predictions[endpoint]
        meta = ENDPOINT_META[endpoint]
        status_class = "status-alert" if result.alert else "status-clear"
        status_text = "达到锁定警报阈值" if result.alert else "未达到锁定警报阈值"
        column.markdown(
            f'<div class="risk-card" style="border-top: 5px solid {meta["color"]};">'
            f'<div class="risk-title"><strong>{meta["label"]}</strong><br>{meta["title"]}</div>'
            f'<div class="risk-prob">{result.locked_probability:.1%}</div>'
            f'<div class="risk-threshold">锁定阈值 {result.threshold:.4%}</div>'
            f'<span class="{status_class}">{status_text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label" style="margin-top:1.6rem;">04 · 局部解释与技术证据</div>', unsafe_allow_html=True)
    st.caption("贡献值在锁定校准 logit 空间中加和。正值提高本次预测 logit，负值降低；这些是预测贡献，不是治疗因果效应。")
    endpoint_tabs = st.tabs(["Hb", "PLT", "WBC/Neut"])
    for tab, endpoint in zip(endpoint_tabs, ("hb", "plt", "wbc_neut"), strict=True):
        with tab:
            result = predictions[endpoint]
            st.plotly_chart(
                waterfall_figure(result, ENDPOINT_META[endpoint]["color"]),
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
            )
            top = sorted(
                result.raw_feature_shap.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:10]
            st.dataframe(
                pd.DataFrame(top, columns=["冻结 raw 特征", "锁定 logit 贡献"]),
                width="stretch",
                hide_index=True,
                column_config={"锁定 logit 贡献": st.column_config.NumberColumn(format="%.6f")},
            )
            with st.expander("查看 raw margin、概率与 SHAP 加和诊断"):
                diagnostics = pd.DataFrame(
                    [
                        ["raw margin", result.raw_margin],
                        ["raw probability", result.raw_probability],
                        ["locked logit", result.locked_logit],
                        ["locked probability", result.locked_probability],
                        ["threshold", result.threshold],
                        ["raw-margin additivity error", result.max_raw_additivity_error],
                        ["locked-logit additivity error", result.max_locked_additivity_error],
                        ["encoded-to-raw aggregation error", result.max_aggregation_error],
                    ],
                    columns=["技术字段", "值"],
                )
                st.dataframe(diagnostics, width="stretch", hide_index=True)

    st.markdown(
        '<p class="tiny-note">本页不保存输入行，不生成含患者信息的 PDF，也不将预测结果写回临床系统。界面中的警报是固定研究阈值的技术状态，不是临床建议。</p>',
        unsafe_allow_html=True,
    )
