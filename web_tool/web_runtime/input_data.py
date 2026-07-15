"""Privacy-aware adapters for one-cycle research feature input."""

from __future__ import annotations

import io
import json
import math
import re
from collections.abc import Sequence

import numpy as np
import pandas as pd


MAX_UPLOAD_BYTES = 2 * 1024 * 1024
SENSITIVE_COLUMN_PATTERN = re.compile(
    r"(?:^|[_\s-])(?:"
    r"name|patient|subject|mrn|medical_record|hospital|admission|visit|"
    r"id_card|identity|phone|mobile|address|email|date|time|dob|birth|"
    r"\u59d3\u540d|\u4f4f\u9662\u53f7|\u75c5\u6848\u53f7|\u8eab\u4efd\u8bc1|\u7535\u8bdd|\u5730\u5740|\u65e5\u671f|\u65f6\u95f4"
    r")(?:$|[_\s-])",
    flags=re.IGNORECASE,
)


class InputValidationError(ValueError):
    """Raised when uploaded input violates schema or privacy constraints."""


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _parse_id_list(value: object, *, column: str) -> object:
    if _is_missing(value):
        return None
    if isinstance(value, (list, tuple, set, np.ndarray)):
        raw_items = list(value)
    elif isinstance(value, str):
        token = value.strip()
        try:
            decoded = json.loads(token)
        except json.JSONDecodeError:
            decoded = [item.strip() for item in token.split(",") if item.strip()]
        raw_items = decoded if isinstance(decoded, list) else [decoded]
    else:
        raw_items = [value]

    parsed: list[int] = []
    for item in raw_items:
        if isinstance(item, bool):
            raise InputValidationError(f"{column} \u53ea\u5141\u8bb8\u975e\u8d1f\u6574\u6570 ID")
        try:
            number = float(item)
        except (TypeError, ValueError) as error:
            raise InputValidationError(f"{column} \u542b\u6709\u65e0\u6cd5\u89e3\u6790\u7684 ID") from error
        if not math.isfinite(number) or not number.is_integer() or number < 0:
            raise InputValidationError(f"{column} \u53ea\u5141\u8bb8\u975e\u8d1f\u6574\u6570 ID")
        parsed.append(int(number))
    return sorted(set(parsed))


def sensitive_columns(columns: Sequence[object]) -> list[str]:
    return sorted(
        str(column)
        for column in columns
        if SENSITIVE_COLUMN_PATTERN.search(str(column).strip())
    )


def prepare_uploaded_frame(
    payload: bytes,
    raw_feature_order: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read one CSV row, reject identity fields, and align to the frozen schema."""

    if not isinstance(payload, bytes) or not payload:
        raise InputValidationError("\u4e0a\u4f20\u6587\u4ef6\u4e3a\u7a7a")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise InputValidationError("\u4e0a\u4f20\u6587\u4ef6\u8d85\u8fc7 2 MB \u9650\u5236")
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except Exception as error:
        raise InputValidationError("CSV \u65e0\u6cd5\u89e3\u6790\uff1b\u8bf7\u4f7f\u7528 UTF-8 \u7f16\u7801") from error
    if len(frame) != 1:
        raise InputValidationError("\u4ec5\u5141\u8bb8\u4e0a\u4f20 1 \u4e2a\u6cbb\u7597\u5468\u671f\u7684\u5355\u884c\u6570\u636e")
    if frame.columns.duplicated().any():
        raise InputValidationError("CSV \u542b\u6709\u91cd\u590d\u5217\u540d")

    blocked = sensitive_columns(frame.columns)
    if blocked:
        raise InputValidationError(
            "\u68c0\u6d4b\u5230\u7591\u4f3c\u8eab\u4efd/\u65e5\u671f\u5b57\u6bb5\uff0c\u5df2\u62d2\u7edd\u5904\u7406\uff1a" + ", ".join(blocked)
        )

    expected = tuple(str(column) for column in raw_feature_order)
    extras = sorted(set(map(str, frame.columns)) - set(expected))
    if extras:
        raise InputValidationError("\u5b58\u5728\u975e\u51bb\u7ed3\u7279\u5f81\u5217\uff1a" + ", ".join(extras))
    missing_columns = [column for column in expected if column not in frame.columns]
    aligned = frame.copy()
    for column in missing_columns:
        aligned[column] = pd.NA
    aligned = aligned.loc[:, expected]
    for column in ("drug_id", "category_id"):
        # Pandas 3 infers CSV text as a string extension dtype, which rejects
        # assigning the parsed list value unless this contract column is object.
        aligned[column] = aligned[column].astype(object)
        aligned.at[aligned.index[0], column] = _parse_id_list(
            aligned.iloc[0][column], column=column
        )

    missing_values = [
        column for column in expected if _is_missing(aligned.iloc[0][column])
    ]
    audit = {
        "source": "uploaded_csv",
        "n_rows": 1,
        "n_expected_features": len(expected),
        "n_missing_columns": len(missing_columns),
        "missing_columns": missing_columns,
        "n_missing_values": len(missing_values),
        "missing_values": missing_values,
        "privacy_check": "passed_no_identity_or_date_columns",
    }
    return aligned, audit


def synthetic_frame(raw_feature_order: Sequence[str], *, profile: str = "neutral") -> pd.DataFrame:
    """Create deterministic, identity-free QA input; values are not a patient record."""

    expected = tuple(str(column) for column in raw_feature_order)
    values: dict[str, object] = {column: np.nan for column in expected}
    common: dict[str, object] = {
        "age": 50.0,
        "gender": "M",
        "is_smoking": 0,
        "is_drinking": 0,
        "abo_blood_type": "Unknown",
        "c_t_stage": 3,
        "c_n_stage": 2,
        "c_m_stage": 0,
        "clinic_stage": 3,
        "is_chemo": 1,
        "is_target": 0,
        "is_immuno": 0,
        "is_rt": 1,
        "cum_chemo": 1.0,
        "cum_target": 0.0,
        "cum_immuno": 0.0,
        "cum_rt": 1.0,
        "is_first_cycle": 1,
        "drug_id": [2, 17],
        "category_id": [2, 5],
    }
    neutral_labs = {
        "base_ALB": 42.0,
        "base_ALT": 24.0,
        "base_AST": 25.0,
        "base_Crea": 72.0,
        "base_Hb": 132.0,
        "base_Hct": 40.0,
        "base_Neut": 3.8,
        "base_PLT": 220.0,
        "base_RBC": 4.6,
        "base_WBC": 6.0,
        "prev_nadir_Hb": 118.0,
        "prev_nadir_Neut": 2.1,
        "prev_nadir_PLT": 165.0,
        "prev_nadir_WBC": 3.7,
    }
    lower_reserve_labs = {
        "base_ALB": 34.0,
        "base_ALT": 45.0,
        "base_AST": 40.0,
        "base_Crea": 88.0,
        "base_Hb": 102.0,
        "base_Hct": 31.0,
        "base_Neut": 1.8,
        "base_PLT": 118.0,
        "base_RBC": 3.5,
        "base_WBC": 3.2,
        "prev_nadir_Hb": 86.0,
        "prev_nadir_Neut": 0.9,
        "prev_nadir_PLT": 72.0,
        "prev_nadir_WBC": 1.7,
        "cum_chemo": 3.0,
        "cum_rt": 2.0,
        "is_first_cycle": 0,
    }
    if profile not in {"neutral", "lower_reserve"}:
        raise InputValidationError(f"\u672a\u77e5\u5408\u6210\u6837\u4f8b\uff1a{profile}")
    values.update(common)
    values.update(neutral_labs)
    if profile == "lower_reserve":
        values.update(lower_reserve_labs)
    return pd.DataFrame([values], columns=expected)


def frame_to_csv(frame: pd.DataFrame) -> bytes:
    serializable = frame.copy()
    for column in ("drug_id", "category_id"):
        serializable[column] = serializable[column].map(
            lambda value: json.dumps(value, ensure_ascii=False)
            if isinstance(value, (list, tuple, set, np.ndarray))
            else value
        )
    return serializable.to_csv(index=False).encode("utf-8")
