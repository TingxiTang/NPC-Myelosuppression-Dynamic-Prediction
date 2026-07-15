"""Session-only, identity-free PDF reporting for frozen Web predictions."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Mapping

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .predictor import EndpointPrediction


ENDPOINTS = ("hb", "plt", "wbc_neut")
ENDPOINT_LABELS = {
    "hb": {"zh": "Grade ≥3 贫血", "en": "Grade ≥3 anemia"},
    "plt": {"zh": "Grade ≥3 血小板减少", "en": "Grade ≥3 thrombocytopenia"},
    "wbc_neut": {
        "zh": "Grade ≥3 白细胞/中性粒细胞减少",
        "en": "Grade ≥3 leukopenia/neutropenia",
    },
}
REPORT_TEXT = {
    "zh": {
        "title": "骨髓抑制风险预测报告",
        "subtitle": "研究用途 · 无身份信息会话报告",
        "summary": "三项风险结果",
        "outcome": "结局",
        "probability": "风险概率",
        "threshold": "锁定阈值",
        "status": "风险状态",
        "high": "达到警报阈值",
        "low": "未达到警报阈值",
        "explanation": "SHAP 瀑布图",
        "other": "其余特征",
        "output": "本次预测",
        "baseline": "模型基线",
        "notice": "本报告仅包含预测结果与特征贡献，不包含姓名、ID、日期或完整输入行。结果仅用于研究和技术验证，不用于临床决策。",
        "causal": "SHAP 表示预测贡献，不代表治疗因果效应。",
    },
    "en": {
        "title": "Bone Marrow Suppression Risk Prediction Report",
        "subtitle": "Research use · Identity-free session report",
        "summary": "Three-outcome risk summary",
        "outcome": "Outcome",
        "probability": "Risk probability",
        "threshold": "Locked threshold",
        "status": "Risk status",
        "high": "Above alert threshold",
        "low": "Below alert threshold",
        "explanation": "SHAP waterfall",
        "other": "Other features",
        "output": "Prediction",
        "baseline": "Model baseline",
        "notice": "This report contains only predictions and feature contributions. It excludes names, IDs, dates, and the complete input row. Results are for research and technical validation only, not clinical decision-making.",
        "causal": "SHAP values are predictive contributions, not causal treatment effects.",
    },
}

_ZH_FONT_NAME = "NotoSansSC-ReportSubset"
_ZH_FONT_PATH = Path(__file__).with_name("assets") / "NotoSansSC-Regular-ReportSubset.ttf"


class _IdentityFreeCanvas(Canvas):
    """Suppress automatically generated timestamps from PDF metadata."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._doc.info._dateFormatter = lambda *_parts: ""


def _font(language: str) -> str:
    if language == "zh":
        try:
            pdfmetrics.getFont(_ZH_FONT_NAME)
        except KeyError:
            pdfmetrics.registerFont(TTFont(_ZH_FONT_NAME, _ZH_FONT_PATH))
        return _ZH_FONT_NAME
    return "Helvetica"


def _waterfall_drawing(
    prediction: EndpointPrediction,
    *,
    language: str,
    font_name: str,
) -> Drawing:
    text = REPORT_TEXT[language]
    ordered = sorted(
        prediction.raw_feature_shap.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    selected = ordered[:8]
    other = sum(value for _, value in ordered[8:])
    rows = [*selected, (text["other"], other)]
    cumulative = [prediction.locked_logit_base]
    for _, contribution in rows:
        cumulative.append(cumulative[-1] + contribution)
    values = [*cumulative, prediction.locked_logit]
    minimum = min(values)
    maximum = max(values)
    span = max(maximum - minimum, 1.0)
    minimum -= span * 0.1
    maximum += span * 0.1

    width = 500
    height = 275
    label_width = 135
    plot_left = label_width + 10
    plot_width = width - plot_left - 12
    row_height = 23
    top = height - 38
    drawing = Drawing(width, height)

    def scale(value: float) -> float:
        return plot_left + (value - minimum) / (maximum - minimum) * plot_width

    drawing.add(
        String(
            0,
            height - 16,
            f'{text["baseline"]}: {prediction.locked_logit_base:.3f}',
            fontName=font_name,
            fontSize=9,
            fillColor=colors.HexColor("#4B5563"),
        )
    )
    current = prediction.locked_logit_base
    previous_y = None
    for index, (name, contribution) in enumerate(rows):
        y = top - index * row_height
        next_value = current + contribution
        x_start = scale(current)
        x_end = scale(next_value)
        if previous_y is not None:
            drawing.add(Line(x_start, previous_y, x_start, y + 5, strokeColor=colors.HexColor("#9CA3AF"), strokeWidth=.6))
        drawing.add(
            String(
                0,
                y,
                str(name)[:25],
                fontName=font_name,
                fontSize=8,
                fillColor=colors.HexColor("#111827"),
            )
        )
        drawing.add(
            Rect(
                min(x_start, x_end),
                y - 3,
                max(abs(x_end - x_start), 1.2),
                10,
                fillColor=colors.HexColor("#FF4B4B" if contribution >= 0 else "#1F9BFF"),
                strokeColor=None,
            )
        )
        drawing.add(
            String(
                min(max(x_end + (3 if contribution >= 0 else -28), plot_left), width - 34),
                y,
                f"{contribution:+.3f}",
                fontName="Helvetica",
                fontSize=7,
                fillColor=colors.HexColor("#374151"),
            )
        )
        current = next_value
        previous_y = y - 3

    output_y = top - len(rows) * row_height
    output_x = scale(prediction.locked_logit)
    drawing.add(Line(output_x, output_y - 3, output_x, top + 12, strokeColor=colors.HexColor("#7C3AED"), strokeWidth=1.2))
    drawing.add(
        String(
            0,
            output_y,
            text["output"],
            fontName=font_name,
            fontSize=8,
            fillColor=colors.HexColor("#111827"),
        )
    )
    drawing.add(
        String(
            min(max(output_x + 4, plot_left), width - 65),
            output_y,
            f"logit {prediction.locked_logit:.3f}",
            fontName="Helvetica",
            fontSize=8,
            fillColor=colors.HexColor("#7C3AED"),
        )
    )
    axis_y = 12
    drawing.add(Line(plot_left, axis_y, width - 12, axis_y, strokeColor=colors.HexColor("#6B7280"), strokeWidth=.7))
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        axis_value = minimum + (maximum - minimum) * fraction
        axis_x = plot_left + plot_width * fraction
        drawing.add(Line(axis_x, axis_y - 2, axis_x, axis_y + 2, strokeColor=colors.HexColor("#6B7280"), strokeWidth=.7))
        drawing.add(String(axis_x - 10, 0, f"{axis_value:.1f}", fontName="Helvetica", fontSize=6.5, fillColor=colors.HexColor("#6B7280")))
    return drawing


def build_pdf_report(
    predictions: Mapping[str, EndpointPrediction],
    *,
    language: str = "zh",
) -> bytes:
    """Create an in-memory PDF without accepting or exporting patient input fields."""

    if language not in REPORT_TEXT:
        raise ValueError("unsupported report language")
    if tuple(predictions) != ENDPOINTS:
        raise ValueError("report requires the frozen three-endpoint prediction order")

    text = REPORT_TEXT[language]
    font_name = _font(language)
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=text["title"],
        author="",
        subject="Identity-free research-use prediction report",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=22,
        leading=29,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#111827"),
        spaceAfter=7 * mm,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=9,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#6B7280"),
        spaceAfter=8 * mm,
    )
    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=14,
        leading=19,
        textColor=colors.HexColor("#1D4ED8"),
        spaceAfter=4 * mm,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#374151"),
    )

    story = [
        Paragraph(text["title"], title_style),
        Paragraph(text["subtitle"], subtitle_style),
        Paragraph(text["summary"], heading_style),
    ]
    table_data = [[text["outcome"], text["probability"], text["threshold"], text["status"]]]
    for endpoint in ENDPOINTS:
        prediction = predictions[endpoint]
        table_data.append(
            [
                ENDPOINT_LABELS[endpoint][language],
                f"{prediction.locked_probability:.1%}",
                f"{prediction.threshold:.2%}",
                text["high"] if prediction.alert else text["low"],
            ]
        )
    summary_table = Table(table_data, colWidths=[62 * mm, 30 * mm, 30 * mm, 48 * mm], repeatRows=1)
    summary_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), .5, colors.HexColor("#9CA3AF")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([summary_table, Spacer(1, 8 * mm), Paragraph(text["notice"], body_style)])

    for endpoint in ENDPOINTS:
        prediction = predictions[endpoint]
        story.extend(
            [
                PageBreak(),
                Paragraph(f'{ENDPOINT_LABELS[endpoint][language]} · {text["explanation"]}', heading_style),
                Paragraph(
                    f'{text["probability"]}: {prediction.locked_probability:.1%} &nbsp;&nbsp; '
                    f'{text["threshold"]}: {prediction.threshold:.2%} &nbsp;&nbsp; '
                    f'{text["status"]}: {text["high"] if prediction.alert else text["low"]}',
                    body_style,
                ),
                Spacer(1, 4 * mm),
                _waterfall_drawing(prediction, language=language, font_name=font_name),
                Spacer(1, 4 * mm),
                Paragraph(text["causal"], body_style),
            ]
        )

    document.build(story, canvasmaker=_IdentityFreeCanvas)
    return buffer.getvalue()
