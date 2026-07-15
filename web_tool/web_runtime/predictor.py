"""Fail-closed inference and native TreeSHAP for the frozen XGBoost bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import pandas as pd
from xgboost import DMatrix, XGBClassifier


def _bootstrap_canonical_imports() -> None:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "final_analysis" / "src" / "encoding.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
    raise RuntimeError("canonical final_analysis runtime is unavailable")


_bootstrap_canonical_imports()

from final_analysis.src.calibration import (  # noqa: E402
    LogisticRecalibrator,
    apply_logistic_recalibrator,
)
from final_analysis.src.encoding import (  # noqa: E402
    ABO_COLUMNS,
    BINARY_COLUMNS,
    FrozenEncoder,
    transform_encoder,
)
from final_analysis.src.preprocess import (  # noqa: E402
    FrozenPreprocessor,
    transform_preprocessor,
)


class ArtifactValidationError(RuntimeError):
    """Raised when frozen asset identity or structure is not exact."""


@dataclass(frozen=True)
class EndpointAssets:
    endpoint: str
    model: XGBClassifier
    encoder: FrozenEncoder
    preprocessor: FrozenPreprocessor
    feature_order: tuple[str, ...]
    calibrator: LogisticRecalibrator
    threshold: float


@dataclass(frozen=True)
class EndpointPrediction:
    endpoint: str
    raw_margin: float
    raw_probability: float
    locked_logit: float
    locked_probability: float
    threshold: float
    alert: bool
    locked_logit_base: float
    raw_feature_shap: Mapping[str, float]
    max_raw_additivity_error: float
    max_locked_additivity_error: float
    max_aggregation_error: float


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"artifact JSON is unreadable: {path.name}") from error


def _require_hash(path: Path, expected: str) -> None:
    if not path.is_file() or path.is_symlink() or _file_sha256(path) != expected:
        raise ArtifactValidationError(f"artifact checksum mismatch: {path.name}")


def _sigmoid(value: float) -> float:
    if value >= 0:
        return float(1.0 / (1.0 + math.exp(-value)))
    exp_value = math.exp(value)
    return float(exp_value / (1.0 + exp_value))


def _raw_mapping(encoder: FrozenEncoder) -> dict[str, tuple[str, ...]]:
    encoded = tuple(encoder.encoded_feature_order)
    encoded_set = set(encoded)
    mapping: dict[str, tuple[str, ...]] = {}
    for raw in encoder.raw_feature_order:
        if raw == "abo_blood_type":
            members = tuple(ABO_COLUMNS)
        elif raw in BINARY_COLUMNS:
            members = (raw, f"{raw}_missing")
        elif raw == "drug_id":
            members = tuple(
                name
                for name in encoded
                if name.startswith("drug_") or name == "drug_mapping_missing"
            )
        elif raw == "category_id":
            members = tuple(
                name
                for name in encoded
                if name.startswith("cat_") or name == "category_mapping_missing"
            )
        else:
            members = (raw,)
        if not members or any(member not in encoded_set for member in members):
            raise ArtifactValidationError(f"encoded-to-raw mapping is incomplete: {raw}")
        mapping[raw] = members
    flattened = [member for members in mapping.values() for member in members]
    if len(flattened) != len(set(flattened)) or set(flattened) != encoded_set:
        raise ArtifactValidationError("encoded-to-raw mapping is not exhaustive")
    return mapping


class FrozenWebPredictor:
    """Load, verify, and score the immutable three-endpoint XGBoost contract."""

    def __init__(self, artifact_root: str | Path, contract_path: str | Path):
        self.artifact_root = Path(artifact_root)
        self.contract_path = Path(contract_path)
        contract = _read_json(self.contract_path)
        if not isinstance(contract, dict):
            raise ArtifactValidationError("artifact contract must be an object")
        self.contract = contract
        self.prediction_tolerance = float(contract["prediction_absolute_tolerance"])
        self.shap_tolerance = float(contract["shap_additivity_absolute_tolerance"])
        self._validate_contract()
        self.selection_lock = self._load_selection_lock()
        self.assets = {
            endpoint: self._load_endpoint(endpoint)
            for endpoint in tuple(str(value) for value in contract["endpoints"])
        }
        raw_orders = {asset.encoder.raw_feature_order for asset in self.assets.values()}
        if len(raw_orders) != 1:
            raise ArtifactValidationError("endpoint raw feature orders disagree")
        self.raw_feature_order = next(iter(raw_orders))
        drug_vocabs = {asset.encoder.drug_vocab for asset in self.assets.values()}
        category_vocabs = {asset.encoder.category_vocab for asset in self.assets.values()}
        if len(drug_vocabs) != 1 or len(category_vocabs) != 1:
            raise ArtifactValidationError("endpoint treatment vocabularies disagree")
        self.drug_vocab = next(iter(drug_vocabs))
        self.category_vocab = next(iter(category_vocabs))
        if len(self.raw_feature_order) != int(contract["raw_feature_count"]):
            raise ArtifactValidationError("raw feature count violates artifact contract")

    def _validate_contract(self) -> None:
        endpoints = tuple(self.contract.get("endpoints", ()))
        if (
            self.contract.get("contract_version") != "research_web_artifact_contract_v1"
            or self.contract.get("selected_family") != "xgboost"
            or endpoints != ("hb", "plt", "wbc_neut")
            or self.contract.get("classification_operator") != ">="
            or self.contract.get("private_assets_allowed") is not False
            or not 0 < self.prediction_tolerance <= 1e-12
            or not 0 < self.shap_tolerance <= 1e-4
        ):
            raise ArtifactValidationError("artifact contract is incomplete or unsafe")

    def _load_selection_lock(self) -> dict[str, object]:
        path = self.artifact_root / "selection_lock" / "selection_lock.json"
        _require_hash(path, str(self.contract["selection_lock_sha256"]))
        payload = _read_json(path)
        endpoints = set(self.contract["endpoints"])
        if (
            not isinstance(payload, dict)
            or payload.get("selected_family") != "xgboost"
            or set(payload.get("selected_pipeline_references", {})) != endpoints
            or set(payload.get("final_calibrators", {})) != endpoints
            or set(payload.get("final_thresholds", {})) != endpoints
        ):
            raise ArtifactValidationError("selection lock is not the frozen XGBoost lock")
        return payload

    def _load_endpoint(self, endpoint: str) -> EndpointAssets:
        references = self.selection_lock["selected_pipeline_references"][endpoint]
        if not isinstance(references, dict):
            raise ArtifactValidationError(f"pipeline reference is invalid: {endpoint}")
        model_root = self.artifact_root / "models" / endpoint
        artifact_refs = references.get("artifacts", {})
        required_names = (
            "encoder.json",
            "preprocessor.json",
            "feature_order.json",
            "model.xgboost.ubj",
        )
        for name in required_names:
            reference = artifact_refs.get(name)
            if not isinstance(reference, dict) or not reference.get("sha256"):
                raise ArtifactValidationError(f"missing frozen asset reference: {endpoint}/{name}")
            _require_hash(model_root / name, str(reference["sha256"]))

        calibrator_path = self.artifact_root / "selection_lock" / "calibrators" / f"{endpoint}.json"
        threshold_path = self.artifact_root / "selection_lock" / "thresholds" / f"{endpoint}.json"
        _require_hash(calibrator_path, str(references["calibrator"]["sha256"]))
        _require_hash(threshold_path, str(references["threshold"]["sha256"]))
        calibrator_payload = _read_json(calibrator_path)
        threshold_payload = _read_json(threshold_path)
        if not isinstance(calibrator_payload, dict) or not isinstance(threshold_payload, dict):
            raise ArtifactValidationError(f"calibration/threshold payload is invalid: {endpoint}")

        frozen_calibrator = self.selection_lock["final_calibrators"][endpoint]
        for key in ("method", "intercept", "slope", "epsilon"):
            if calibrator_payload.get(key) != frozen_calibrator.get(key):
                raise ArtifactValidationError(f"calibrator disagrees with selection lock: {endpoint}")
        frozen_threshold = float(self.selection_lock["final_thresholds"][endpoint]["threshold"])
        if (
            threshold_payload.get("classification_operator") != ">="
            or float(threshold_payload.get("threshold")) != frozen_threshold
        ):
            raise ArtifactValidationError(f"threshold disagrees with selection lock: {endpoint}")

        encoder_payload = _read_json(model_root / "encoder.json")
        preprocessor_payload = _read_json(model_root / "preprocessor.json")
        feature_order_payload = _read_json(model_root / "feature_order.json")
        if not isinstance(encoder_payload, dict) or not isinstance(preprocessor_payload, dict):
            raise ArtifactValidationError(f"encoder/preprocessor payload is invalid: {endpoint}")
        if not isinstance(feature_order_payload, list) or not feature_order_payload:
            raise ArtifactValidationError(f"feature order payload is invalid: {endpoint}")
        encoder = FrozenEncoder.from_dict(encoder_payload)
        preprocessor = FrozenPreprocessor.from_dict(preprocessor_payload)
        feature_order = tuple(str(value) for value in feature_order_payload)
        expected_encoded = int(self.contract["encoded_feature_count"])
        if (
            preprocessor.algorithm != "xgboost"
            or encoder.encoded_feature_order != preprocessor.input_feature_order
            or preprocessor.output_feature_order != feature_order
            or len(feature_order) != expected_encoded
            or int(references.get("n_features", -1)) != expected_encoded
            or int(references.get("n_raw_features", -1)) != len(encoder.raw_feature_order)
        ):
            raise ArtifactValidationError(f"frozen preprocessing contract drift: {endpoint}")

        model = XGBClassifier()
        model.load_model(model_root / "model.xgboost.ubj")
        booster_names = tuple(model.get_booster().feature_names or ())
        if booster_names != feature_order:
            raise ArtifactValidationError(f"model feature order drift: {endpoint}")
        return EndpointAssets(
            endpoint=endpoint,
            model=model,
            encoder=encoder,
            preprocessor=preprocessor,
            feature_order=feature_order,
            calibrator=LogisticRecalibrator.from_dict(calibrator_payload),
            threshold=frozen_threshold,
        )

    def _matrix(self, asset: EndpointAssets, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame) or len(frame) != 1:
            raise ValueError("prediction input must contain exactly one row")
        raw = frame.loc[:, asset.encoder.raw_feature_order]
        encoded = transform_encoder(asset.encoder, raw)
        matrix = transform_preprocessor(asset.preprocessor, encoded)
        return matrix.loc[:, asset.feature_order]

    def predict_endpoint(self, endpoint: str, frame: pd.DataFrame) -> EndpointPrediction:
        asset = self.assets[endpoint]
        matrix = self._matrix(asset, frame)
        booster = asset.model.get_booster()
        dmatrix = DMatrix(matrix, feature_names=list(asset.feature_order))
        raw_margin = float(booster.predict(dmatrix, output_margin=True, validate_features=True)[0])
        raw_probability = float(asset.model.predict_proba(matrix)[0, 1])
        locked_probability = float(
            apply_logistic_recalibrator(asset.calibrator, [raw_probability])[0]
        )
        clipped = min(max(raw_probability, asset.calibrator.epsilon), 1 - asset.calibrator.epsilon)
        locked_logit = float(
            asset.calibrator.intercept
            + asset.calibrator.slope * math.log(clipped / (1.0 - clipped))
        )
        if abs(_sigmoid(locked_logit) - locked_probability) > self.prediction_tolerance:
            raise RuntimeError(f"locked probability formula drift: {endpoint}")

        contributions = np.asarray(
            booster.predict(dmatrix, pred_contribs=True, validate_features=True),
            dtype=float,
        )
        if contributions.shape != (1, len(asset.feature_order) + 1):
            raise RuntimeError(f"native TreeSHAP shape drift: {endpoint}")
        raw_error = float(abs(contributions[0].sum() - raw_margin))
        calibrated = asset.calibrator.slope * contributions[0, :-1]
        locked_base = float(
            asset.calibrator.intercept + asset.calibrator.slope * contributions[0, -1]
        )
        locked_error = float(abs(calibrated.sum() + locked_base - locked_logit))
        mapping = _raw_mapping(asset.encoder)
        positions = {name: index for index, name in enumerate(asset.feature_order)}
        raw_shap = {
            raw: float(sum(calibrated[positions[member]] for member in members))
            for raw, members in mapping.items()
        }
        aggregation_error = float(abs(sum(raw_shap.values()) - calibrated.sum()))
        if (
            raw_error > self.shap_tolerance
            or locked_error > self.shap_tolerance
            or aggregation_error > min(self.shap_tolerance, 1e-10)
        ):
            raise RuntimeError(f"SHAP additivity drift: {endpoint}")

        return EndpointPrediction(
            endpoint=endpoint,
            raw_margin=raw_margin,
            raw_probability=raw_probability,
            locked_logit=locked_logit,
            locked_probability=locked_probability,
            threshold=asset.threshold,
            alert=locked_probability >= asset.threshold,
            locked_logit_base=locked_base,
            raw_feature_shap=raw_shap,
            max_raw_additivity_error=raw_error,
            max_locked_additivity_error=locked_error,
            max_aggregation_error=aggregation_error,
        )

    def predict_all(self, frame: pd.DataFrame) -> dict[str, EndpointPrediction]:
        return {
            endpoint: self.predict_endpoint(endpoint, frame)
            for endpoint in tuple(self.contract["endpoints"])
        }

    def technical_summary(self) -> dict[str, object]:
        return {
            "bundle_id": self.contract["independent_bundle_id"],
            "selection_lock_sha256": self.contract["selection_lock_sha256"],
            "model_family": "XGBoost",
            "raw_features": len(self.raw_feature_order),
            "encoded_features": int(self.contract["encoded_feature_count"]),
            "classification_operator": ">=",
            "thresholds": {
                endpoint: asset.threshold for endpoint, asset in self.assets.items()
            },
            "shap_output_space": "locked_calibrated_logit",
        }
