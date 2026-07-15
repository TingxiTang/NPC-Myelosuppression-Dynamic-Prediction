"""Leakage-safe logistic recalibration and exact sensitivity thresholds."""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Mapping

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


CALIBRATION_EPSILON = 1e-6
CALIBRATION_MAX_ITER = 2_000
CALIBRATION_TOLERANCE = 1e-10
THRESHOLD_FLOAT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class LogisticRecalibrator:
    intercept: float
    slope: float
    epsilon: float = CALIBRATION_EPSILON

    def __post_init__(self) -> None:
        if not math.isfinite(self.intercept):
            raise ValueError("calibration intercept must be finite")
        if not math.isfinite(self.slope) or self.slope <= 0:
            raise ValueError("calibration slope must be positive")
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon < 0.5:
            raise ValueError("calibration epsilon must lie in (0,0.5)")

    def to_dict(self) -> dict[str, float | str]:
        return {
            "method": "logistic_recalibration",
            "intercept": self.intercept,
            "slope": self.slope,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LogisticRecalibrator":
        if payload.get("method") != "logistic_recalibration":
            raise ValueError("calibration method is invalid")
        return cls(
            intercept=float(payload["intercept"]),
            slope=float(payload["slope"]),
            epsilon=float(payload["epsilon"]),
        )


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    sensitivity_target: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    alert_rate: float
    false_positives_per_100: float
    number_needed_to_evaluate: float
    tp: int
    fn: int
    tn: int
    fp: int


def _probability_array(values, *, name: str = "probability") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} must lie in [0,1]")
    return array


def _outcome_array(values, *, require_both: bool = True) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("outcome must be a nonempty one-dimensional array")
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("outcome must contain only 0/1") from error
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0, 1]).all():
        raise ValueError("outcome must contain only 0/1")
    outcome = numeric.astype("int8")
    if require_both and set(outcome.tolist()) != {0, 1}:
        raise ValueError("outcome must contain both classes")
    return outcome


def _logit_probability(probability: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(probability, epsilon, 1 - epsilon)
    return np.log(clipped / (1 - clipped))


def fit_logistic_recalibrator(
    raw_probability,
    outcome,
    *,
    epsilon: float = CALIBRATION_EPSILON,
) -> LogisticRecalibrator:
    probability = _probability_array(raw_probability, name="raw probability")
    labels = _outcome_array(outcome)
    if len(probability) != len(labels):
        raise ValueError("raw probability and outcome length mismatch")
    if not math.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ValueError("calibration epsilon must lie in (0,0.5)")

    logit = _logit_probability(probability, epsilon).reshape(-1, 1)
    model = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=CALIBRATION_MAX_ITER,
        tol=CALIBRATION_TOLERANCE,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(logit, labels)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise ValueError("logistic recalibration did not converge")
    if int(model.n_iter_[0]) >= CALIBRATION_MAX_ITER:
        raise ValueError("logistic recalibration did not converge")
    return LogisticRecalibrator(
        intercept=float(model.intercept_[0]),
        slope=float(model.coef_[0, 0]),
        epsilon=float(epsilon),
    )


def apply_logistic_recalibrator(
    calibrator: LogisticRecalibrator, raw_probability
) -> np.ndarray:
    if not isinstance(calibrator, LogisticRecalibrator):
        raise TypeError("calibrator must be a LogisticRecalibrator")
    probability = _probability_array(raw_probability, name="raw probability")
    logit = _logit_probability(probability, calibrator.epsilon)
    calibrated = expit(calibrator.intercept + calibrator.slope * logit)
    if not np.isfinite(calibrated).all() or np.any(
        (calibrated < 0) | (calibrated > 1)
    ):
        raise ValueError("calibrated probability is invalid")
    return np.asarray(calibrated, dtype=float)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def select_exact_threshold(
    outcome,
    probability,
    sensitivity_target: float,
) -> ThresholdResult:
    labels = _outcome_array(outcome)
    risk = _probability_array(probability)
    if len(labels) != len(risk):
        raise ValueError("outcome and probability length mismatch")
    if (
        isinstance(sensitivity_target, bool)
        or not isinstance(sensitivity_target, (int, float))
        or not math.isfinite(float(sensitivity_target))
        or not 0 < float(sensitivity_target) <= 1
    ):
        raise ValueError("sensitivity_target must lie in (0,1]")
    target = float(sensitivity_target)

    candidates = np.unique(np.concatenate(([0.0, 1.0], risk)))
    best: tuple[tuple[float, float], ThresholdResult] | None = None
    positive = labels == 1
    negative = ~positive
    for threshold in candidates:
        predicted = risk >= threshold
        tp = int(np.sum(positive & predicted))
        fn = int(np.sum(positive & ~predicted))
        tn = int(np.sum(negative & ~predicted))
        fp = int(np.sum(negative & predicted))
        sensitivity = _safe_ratio(tp, tp + fn)
        if sensitivity + THRESHOLD_FLOAT_TOLERANCE < target:
            continue
        specificity = _safe_ratio(tn, tn + fp)
        ppv = _safe_ratio(tp, tp + fp)
        npv = _safe_ratio(tn, tn + fn)
        alerts = tp + fp
        result = ThresholdResult(
            threshold=float(threshold),
            sensitivity_target=target,
            sensitivity=sensitivity,
            specificity=specificity,
            ppv=ppv,
            npv=npv,
            alert_rate=_safe_ratio(alerts, len(labels)),
            false_positives_per_100=100 * _safe_ratio(fp, len(labels)),
            number_needed_to_evaluate=(
                float(alerts / tp) if tp else float("inf")
            ),
            tp=tp,
            fn=fn,
            tn=tn,
            fp=fp,
        )
        key = (specificity, float(threshold))
        if best is None or key > best[0]:
            best = (key, result)
    if best is None:
        raise ValueError("no threshold satisfies the sensitivity target")
    return best[1]
