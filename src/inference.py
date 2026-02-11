import pandas as pd
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression

current_working_dir = Path.cwd()

# Model paths
model_dir = current_working_dir.parent / "models"
models_paths = {
    'Hb': f'{model_dir}/Hb_LightGBM_Calibrated.joblib',
    'PLT': f'{model_dir}/PLT_LightGBM_Calibrated.joblib',
    'WBC_Neut': f'{model_dir}/WBC_Neut_LightGBM_Calibrated.joblib'
}
LGBM_features = f'{model_dir}/lightgbm_feature_names.joblib'
# Scaler path
scaler_path = f'{model_dir}/scaler_continuous.joblib'
scaler_features_path = f'{model_dir}/scaler_continuous_features.joblib'

scaler = joblib.load(scaler_path)
scaler_feature_names = joblib.load(scaler_features_path)
lgbm_feature_names = joblib.load(LGBM_features)

class PlattScalingCalibrator:
    def __init__(self, base_model):
        self.base_model = base_model
        self.platt_lr = LogisticRegression(max_iter=1000)
    
    def fit(self, X, y):
        if hasattr(self.base_model, "predict_proba"):
            raw_probs = self.base_model.predict_proba(X)[:, 1]
        else:
            raw_probs = self.base_model.predict(X)
        self.platt_lr.fit(raw_probs.reshape(-1, 1), y)
        return self
    
    def predict_proba(self, X):
        if hasattr(self.base_model, "predict_proba"):
            raw_probs = self.base_model.predict_proba(X)[:, 1]
        else:
            raw_probs = self.base_model.predict(X)
        calibrated_probs = self.platt_lr.predict_proba(raw_probs.reshape(-1, 1))
        return calibrated_probs

def align_features(X, feature_names):
    return X.reindex(columns=feature_names)

# X = pd.read_parquet('path_to_input_data.parquet') # Load your input data here
model = joblib.load(models_paths['Hb']) # PLT, WBC_Neut
X_use = align_features(X, lgbm_feature_names)
prob = model.predict(X_use)