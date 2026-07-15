"""Algorithm-specific preprocessing fitted only on training partitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PREPROCESSOR_VERSION = "algorithm_preprocessor_v1"
SUPPORTED_ALGORITHMS = ("lr", "xgboost", "lightgbm", "tabpfn")
SUPPORTED_FAMILIES = {
    "continuous",
    "binary",
    "ordinal",
    "abo_onehot",
    "drug_multihot",
    "category_multihot",
    "missing_indicator",
}


@dataclass(frozen=True)
class FrozenPreprocessor:
    algorithm: str
    input_feature_order: tuple[str, ...]
    output_feature_order: tuple[str, ...]
    continuous_columns: tuple[str, ...]
    numeric_medians: dict[str, float]
    lr_missing_indicator_columns: tuple[str, ...]
    scaler_mean: dict[str, float]
    scaler_scale: dict[str, float]
    feature_families: tuple[tuple[str, str], ...]
    preprocessor_version: str = PREPROCESSOR_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "input_feature_order": list(self.input_feature_order),
            "output_feature_order": list(self.output_feature_order),
            "continuous_columns": list(self.continuous_columns),
            "numeric_medians": dict(self.numeric_medians),
            "lr_missing_indicator_columns": list(
                self.lr_missing_indicator_columns
            ),
            "scaler_mean": dict(self.scaler_mean),
            "scaler_scale": dict(self.scaler_scale),
            "feature_families": [list(item) for item in self.feature_families],
            "preprocessor_version": self.preprocessor_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FrozenPreprocessor":
        return cls(
            algorithm=str(payload["algorithm"]),
            input_feature_order=tuple(
                str(x) for x in payload["input_feature_order"]
            ),
            output_feature_order=tuple(
                str(x) for x in payload["output_feature_order"]
            ),
            continuous_columns=tuple(
                str(x) for x in payload["continuous_columns"]
            ),
            numeric_medians={
                str(key): float(value)
                for key, value in dict(payload["numeric_medians"]).items()
            },
            lr_missing_indicator_columns=tuple(
                str(x) for x in payload["lr_missing_indicator_columns"]
            ),
            scaler_mean={
                str(key): float(value)
                for key, value in dict(payload["scaler_mean"]).items()
            },
            scaler_scale={
                str(key): float(value)
                for key, value in dict(payload["scaler_scale"]).items()
            },
            feature_families=tuple(
                (str(item[0]), str(item[1]))
                for item in payload["feature_families"]
            ),
            preprocessor_version=str(
                payload.get("preprocessor_version", PREPROCESSOR_VERSION)
            ),
        )


def _validate_schema(frame: pd.DataFrame, expected: tuple[str, ...]) -> None:
    actual = tuple(str(column) for column in frame.columns)
    if len(actual) != len(set(actual)) or actual != expected:
        raise ValueError(
            f"input feature order mismatch: expected {expected!r}, got {actual!r}"
        )


def _validate_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    numeric: dict[str, pd.Series] = {}
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"non-numeric encoded feature: {column}")
        finite = converted.dropna().map(np.isfinite)
        if not finite.all():
            raise ValueError(f"infinite value in encoded feature: {column}")
        numeric[str(column)] = converted.astype(float)
    return pd.DataFrame(numeric, index=frame.index)


def _normalize_families(
    order: tuple[str, ...], feature_families: Mapping[str, str]
) -> tuple[tuple[str, str], ...]:
    if tuple(feature_families.keys()) != order:
        raise ValueError(
            "feature families must contain every input column in input order"
        )
    normalized = tuple((column, str(feature_families[column])) for column in order)
    unknown = [family for _, family in normalized if family not in SUPPORTED_FAMILIES]
    if unknown:
        raise ValueError(f"unknown feature families: {sorted(set(unknown))}")
    return normalized


def _validate_missing_contract(
    numeric: pd.DataFrame, families: Mapping[str, str]
) -> None:
    for column in numeric.columns:
        if not numeric[column].isna().any():
            continue
        family = families[column]
        if family == "continuous":
            continue
        if family == "binary" and f"{column}_missing" in numeric.columns:
            continue
        raise ValueError(
            f"missing encoded values are unsupported for {family} feature {column}"
        )


def fit_preprocessor(
    algorithm: str,
    train_encoded: pd.DataFrame,
    feature_families: Mapping[str, str],
) -> FrozenPreprocessor:
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"unsupported algorithm: {algorithm!r}")
    if train_encoded.empty:
        raise ValueError("training frame cannot be empty")
    order = tuple(str(column) for column in train_encoded.columns)
    if len(order) != len(set(order)):
        raise ValueError("input feature order contains duplicate columns")
    families_tuple = _normalize_families(order, feature_families)
    families = dict(families_tuple)
    numeric = _validate_numeric(train_encoded)
    _validate_missing_contract(numeric, families)
    continuous = tuple(
        column for column in order if families[column] == "continuous"
    )
    if not continuous:
        raise ValueError("at least one continuous feature is required")
    all_missing = [column for column in continuous if numeric[column].isna().all()]
    if all_missing:
        raise ValueError(f"continuous training columns are all missing: {all_missing}")

    medians = {
        column: float(numeric[column].median(skipna=True)) for column in continuous
    }
    scaler_input = numeric.loc[:, continuous].copy()
    if algorithm == "lr":
        scaler_input = scaler_input.fillna(medians)
        indicators = tuple(
            f"{column}_missing"
            for column in continuous
            if numeric[column].isna().any()
        )
        collisions = set(indicators) & set(order)
        if collisions:
            raise ValueError(
                f"LR continuous missing indicators collide with input columns: {sorted(collisions)}"
            )
    else:
        medians = {}
        indicators = ()

    scaler = StandardScaler()
    scaler.fit(scaler_input)
    means = {
        column: float(value) for column, value in zip(continuous, scaler.mean_)
    }
    scales = {
        column: float(value) for column, value in zip(continuous, scaler.scale_)
    }
    if any(not math.isfinite(value) for value in (*means.values(), *scales.values())):
        raise ValueError("non-finite scaler parameters")
    return FrozenPreprocessor(
        algorithm=algorithm,
        input_feature_order=order,
        output_feature_order=order + indicators,
        continuous_columns=continuous,
        numeric_medians=medians,
        lr_missing_indicator_columns=indicators,
        scaler_mean=means,
        scaler_scale=scales,
        feature_families=families_tuple,
    )


def transform_preprocessor(
    preprocessor: FrozenPreprocessor, frame: pd.DataFrame
) -> pd.DataFrame:
    _validate_schema(frame, preprocessor.input_feature_order)
    numeric = _validate_numeric(frame)
    families = dict(preprocessor.feature_families)
    _validate_missing_contract(numeric, families)
    out = numeric.copy()

    if preprocessor.algorithm == "lr":
        for indicator in preprocessor.lr_missing_indicator_columns:
            source = indicator.removesuffix("_missing")
            out[indicator] = numeric[source].isna().astype("int8")
        for column, median in preprocessor.numeric_medians.items():
            out[column] = out[column].fillna(median)
        for column, family in families.items():
            if family == "binary" and out[column].isna().any():
                companion = f"{column}_missing"
                if companion not in out.columns:
                    raise ValueError(
                        f"binary feature {column} lacks its missing indicator"
                    )
                out[column] = out[column].fillna(0.0)

    for column in preprocessor.continuous_columns:
        out[column] = (
            out[column] - preprocessor.scaler_mean[column]
        ) / preprocessor.scaler_scale[column]
    out = out.loc[:, preprocessor.output_feature_order]
    if preprocessor.algorithm == "lr" and out.isna().any().any():
        missing = out.columns[out.isna().any()].tolist()
        raise ValueError(f"LR preprocessing left missing values: {missing}")
    return out
