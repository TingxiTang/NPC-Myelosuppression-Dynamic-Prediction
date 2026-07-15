"""Frozen, training-only clinical feature encoding.

The encoder deliberately contains no outcome-dependent logic.  It records the
raw and encoded column orders plus train-fitted multi-value vocabularies so the
same contract can be reused for tuning, independent validation and deployment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
import unicodedata

import numpy as np
import pandas as pd


ENCODER_VERSION = "frozen_clinical_encoder_v1"
ABO_COLUMNS = ("ABO_A", "ABO_B", "ABO_AB", "ABO_O", "ABO_Unknown")
BINARY_COLUMNS = (
    "gender",
    "is_smoking",
    "is_drinking",
    "is_chemo",
    "is_target",
    "is_immuno",
    "is_rt",
    "is_first_cycle",
)
STAGE_COLUMNS = ("c_t_stage", "c_n_stage", "c_m_stage", "clinic_stage")
GENDER_MAP = {
    "女": 0.0,
    "F": 0.0,
    "FEMALE": 0.0,
    "男": 1.0,
    "M": 1.0,
    "MALE": 1.0,
}
BOOL_MAP = {
    "否": 0.0,
    "NO": 0.0,
    "N": 0.0,
    "FALSE": 0.0,
    "0": 0.0,
    "是": 1.0,
    "YES": 1.0,
    "Y": 1.0,
    "TRUE": 1.0,
    "1": 1.0,
}
VALID_FAMILIES = {
    "continuous",
    "binary",
    "ordinal",
    "abo_onehot",
    "drug_multihot",
    "category_multihot",
    "missing_indicator",
}


@dataclass(frozen=True)
class FrozenEncoder:
    raw_feature_order: tuple[str, ...]
    encoded_feature_order: tuple[str, ...]
    feature_families: tuple[tuple[str, str], ...]
    drug_vocab: tuple[int, ...]
    category_vocab: tuple[int, ...]
    encoder_version: str = ENCODER_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_feature_order": list(self.raw_feature_order),
            "encoded_feature_order": list(self.encoded_feature_order),
            "feature_families": [list(item) for item in self.feature_families],
            "drug_vocab": list(self.drug_vocab),
            "category_vocab": list(self.category_vocab),
            "encoder_version": self.encoder_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FrozenEncoder":
        return cls(
            raw_feature_order=tuple(str(x) for x in payload["raw_feature_order"]),
            encoded_feature_order=tuple(
                str(x) for x in payload["encoded_feature_order"]
            ),
            feature_families=tuple(
                (str(item[0]), str(item[1]))
                for item in payload["feature_families"]
            ),
            drug_vocab=tuple(int(x) for x in payload["drug_vocab"]),
            category_vocab=tuple(int(x) for x in payload["category_vocab"]),
            encoder_version=str(payload.get("encoder_version", ENCODER_VERSION)),
        )


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


def _normalize_token(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip().upper()


def _parse_binary(value: object, *, gender: bool, column: str) -> float:
    if _is_missing(value):
        return float("nan")
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return float(value)
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)) and float(value) in (0.0, 1.0):
            return float(value)
    mapping = GENDER_MAP if gender else BOOL_MAP
    token = _normalize_token(value)
    if token in mapping:
        return mapping[token]
    raise ValueError(f"unknown binary value for {column}: {value!r}")


def _parse_stage(value: object, column: str) -> int:
    if _is_missing(value):
        return -1
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        number = int(value)
    elif isinstance(value, (float, np.floating)) and math.isfinite(float(value)) and float(value).is_integer():
        number = int(value)
    else:
        token = _normalize_token(value).replace("期", "")
        if column == "clinic_stage":
            token = token.replace("CLINICAL", "").replace("STAGE", "").strip()
            match = re.fullmatch(r"(IV|III|II|I|[1-4])(?:[ABC])?", token)
            if not match:
                return -1
            number = {"I": 1, "II": 2, "III": 3, "IV": 4}.get(
                match.group(1), int(match.group(1)) if match.group(1).isdigit() else -1
            )
        else:
            letter = {"c_t_stage": "T", "c_n_stage": "N", "c_m_stage": "M"}[column]
            match = re.fullmatch(rf"(?:C)?{letter}([0-9])(?:[ABC])?", token)
            if not match:
                return -1
            number = int(match.group(1))
    maximum = {"c_t_stage": 4, "c_n_stage": 3, "c_m_stage": 1, "clinic_stage": 4}[column]
    minimum = 1 if column == "clinic_stage" else 0
    return number if minimum <= number <= maximum else -1


def _parse_abo(value: object) -> str:
    if _is_missing(value):
        return "Unknown"
    token = _normalize_token(value).replace("型", "")
    return token if token in {"A", "B", "AB", "O"} else "Unknown"


def _parse_id_container(value: object, *, column: str) -> tuple[tuple[int, ...], bool]:
    if _is_missing(value):
        return (), True
    if not isinstance(value, (list, tuple, set, np.ndarray)):
        raise ValueError(f"{column} must be a container of non-negative integers")
    parsed: list[int] = []
    for item in value:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f"{column} IDs must be non-negative integers")
        if isinstance(item, (int, np.integer)):
            parsed_item = int(item)
        elif isinstance(item, (float, np.floating)) and math.isfinite(float(item)) and float(item).is_integer():
            parsed_item = int(item)
        else:
            raise ValueError(f"{column} IDs must be non-negative integers")
        if parsed_item < 0:
            raise ValueError(f"{column} IDs must be non-negative integers")
        parsed.append(parsed_item)
    return tuple(sorted(set(parsed))), False


def _continuous_column(column: str) -> bool:
    return column == "age" or column.startswith(("base_", "prev_nadir_", "cum_"))


def _validate_raw_order(frame: pd.DataFrame, expected: Sequence[str]) -> None:
    actual = tuple(str(column) for column in frame.columns)
    if len(actual) != len(set(actual)) or actual != tuple(expected):
        raise ValueError(
            f"raw feature order mismatch: expected {tuple(expected)!r}, got {actual!r}"
        )


def _vocabulary(series: pd.Series, *, column: str) -> tuple[int, ...]:
    values: set[int] = set()
    for value in series:
        parsed, _ = _parse_id_container(value, column=column)
        values.update(parsed)
    return tuple(sorted(values))


def fit_encoder(
    train: pd.DataFrame, raw_feature_order: Sequence[str]
) -> FrozenEncoder:
    if train.empty:
        raise ValueError("training frame cannot be empty")
    raw_order = tuple(str(column) for column in raw_feature_order)
    if not raw_order or len(raw_order) != len(set(raw_order)):
        raise ValueError("raw feature order must be non-empty and unique")
    _validate_raw_order(train, raw_order)
    unsupported = [
        column
        for column in raw_order
        if not (
            _continuous_column(column)
            or column in BINARY_COLUMNS
            or column in STAGE_COLUMNS
            or column in {"abo_blood_type", "drug_id", "category_id"}
        )
    ]
    if unsupported:
        raise ValueError(f"unsupported raw feature columns: {unsupported}")
    drug_vocab = _vocabulary(train["drug_id"], column="drug_id") if "drug_id" in raw_order else ()
    category_vocab = _vocabulary(train["category_id"], column="category_id") if "category_id" in raw_order else ()

    encoded: list[str] = []
    families: list[tuple[str, str]] = []
    for column in raw_order:
        if column in {"drug_id", "category_id"}:
            continue
        if column == "abo_blood_type":
            encoded.extend(ABO_COLUMNS)
            families.extend((name, "abo_onehot") for name in ABO_COLUMNS)
        elif column in BINARY_COLUMNS:
            encoded.extend((column, f"{column}_missing"))
            families.extend(((column, "binary"), (f"{column}_missing", "missing_indicator")))
        elif column in STAGE_COLUMNS:
            encoded.append(column)
            families.append((column, "ordinal"))
        else:
            encoded.append(column)
            families.append((column, "continuous"))
    if "drug_id" in raw_order:
        names = tuple(f"drug_{value}" for value in drug_vocab) + (
            "drug_UNK", "drug_mapping_missing"
        )
        encoded.extend(names)
        families.extend(
            (name, "missing_indicator" if name == "drug_mapping_missing" else "drug_multihot")
            for name in names
        )
    if "category_id" in raw_order:
        names = tuple(f"cat_{value}" for value in category_vocab) + (
            "cat_UNK", "category_mapping_missing"
        )
        encoded.extend(names)
        families.extend(
            (name, "missing_indicator" if name == "category_mapping_missing" else "category_multihot")
            for name in names
        )
    if len(encoded) != len(set(encoded)):
        raise ValueError("encoded feature names are not unique")
    if any(family not in VALID_FAMILIES for _, family in families):
        raise AssertionError("invalid encoded feature family")
    return FrozenEncoder(
        raw_feature_order=raw_order,
        encoded_feature_order=tuple(encoded),
        feature_families=tuple(families),
        drug_vocab=drug_vocab,
        category_vocab=category_vocab,
    )


def _numeric_series(series: pd.Series, column: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    invalid = series.map(lambda value: not _is_missing(value)) & converted.isna()
    if invalid.any():
        raise ValueError(f"non-numeric value in continuous feature {column}")
    finite = converted.dropna().map(np.isfinite)
    if not finite.all():
        raise ValueError(f"infinite value in continuous feature {column}")
    return converted.astype(float)


def _multihot_columns(
    series: pd.Series,
    *,
    column: str,
    prefix: str,
    vocab: tuple[int, ...],
    index: pd.Index,
) -> dict[str, pd.Series]:
    rows = [_parse_id_container(value, column=column) for value in series]
    known = set(vocab)
    result = {
        f"{prefix}_{value}": pd.Series(
            [int(value in ids) for ids, _ in rows], index=index, dtype="int8"
        )
        for value in vocab
    }
    result[f"{prefix}_UNK"] = pd.Series(
        [int(any(value not in known for value in ids)) for ids, _ in rows],
        index=index,
        dtype="int8",
    )
    missing_name = "drug_mapping_missing" if prefix == "drug" else "category_mapping_missing"
    result[missing_name] = pd.Series(
        [int(missing) for _, missing in rows], index=index, dtype="int8"
    )
    return result


def transform_encoder(encoder: FrozenEncoder, frame: pd.DataFrame) -> pd.DataFrame:
    _validate_raw_order(frame, encoder.raw_feature_order)
    columns: dict[str, pd.Series] = {}
    for column in encoder.raw_feature_order:
        series = frame[column]
        if column in {"drug_id", "category_id"}:
            continue
        if column == "abo_blood_type":
            parsed = series.map(_parse_abo)
            for name in ABO_COLUMNS:
                label = name.removeprefix("ABO_")
                columns[name] = parsed.eq(label).astype("int8")
        elif column in BINARY_COLUMNS:
            parsed = series.map(
                lambda value: _parse_binary(value, gender=column == "gender", column=column)
            ).astype(float)
            columns[column] = parsed
            columns[f"{column}_missing"] = parsed.isna().astype("int8")
        elif column in STAGE_COLUMNS:
            columns[column] = series.map(lambda value: _parse_stage(value, column)).astype("int8")
        else:
            columns[column] = _numeric_series(series, column)
    if "drug_id" in encoder.raw_feature_order:
        columns.update(
            _multihot_columns(
                frame["drug_id"], column="drug_id", prefix="drug",
                vocab=encoder.drug_vocab, index=frame.index
            )
        )
    if "category_id" in encoder.raw_feature_order:
        columns.update(
            _multihot_columns(
                frame["category_id"], column="category_id", prefix="cat",
                vocab=encoder.category_vocab, index=frame.index
            )
        )
    out = pd.DataFrame(columns, index=frame.index)
    actual = tuple(out.columns)
    if actual != encoder.encoded_feature_order:
        raise AssertionError(
            f"encoded feature order drift: expected {encoder.encoded_feature_order!r}, got {actual!r}"
        )
    return out
