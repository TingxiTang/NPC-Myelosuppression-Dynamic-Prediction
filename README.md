# Dynamic Prediction of Treatment-Related Myelosuppression in NPC

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://myelosuppression-pred.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository accompanies the study **“Dynamic prediction of treatment-related myelosuppression in patients with nasopharyngeal carcinoma using routine longitudinal data: a multicentre real-world study with nested prospective clinical trial validation.”**

## Research-use Web prototype

The current Streamlit app is a frozen technical implementation of three XGBoost 3.3.0 endpoints:

- Grade ≥3 anemia (Hb);
- Grade ≥3 thrombocytopenia (PLT);
- Grade ≥3 leukopenia/neutropenia (WBC/Neut).

It uses the frozen 106-raw-feature to 253-encoded-feature pipeline, endpoint-specific preprocessors, logistic recalibrators, thresholds, and local TreeSHAP explanations. The streamlined interface supports Chinese and English, a calibrated-logit SHAP waterfall, and an identity-free in-memory PDF report. Model identity and asset checksums are verified at startup.

**This prototype is for research and technical validation only.** It has not undergone prospective clinical impact evaluation and must not be used for diagnosis, treatment selection, dose adjustment, or replacement of clinical judgment. SHAP values describe predictive contributions and must not be interpreted as treatment effects.

Access: <https://myelosuppression-pred.streamlit.app/>

## Deployment coordinates

- Repository: `TingxiTang/NPC-Myelosuppression-Dynamic-Prediction`
- Branch: `main`
- Entrypoint: `web_tool/app.py`
- Python: 3.12
- Dependencies: `web_tool/requirements.txt`
- Streamlit configuration: `.streamlit/config.toml`

The deployed app reads only the audited runtime and frozen assets under `web_tool/`. Other repository directories are not loaded by the current Web application.

## Run the Web prototype locally

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r web_tool/requirements.txt
.venv/bin/python -m streamlit run web_tool/app.py
```

The application accepts one treatment-cycle row at a time and rejects identity, date/time, free-text, and non-contract fields. Two identity-free synthetic examples are included in the interface for technical testing.

## Data boundary

Raw clinical datasets are not distributed for public Web deployment. The audited `web_tool/` package contains no patient-level rows, sentinel rows, secrets, or machine-specific absolute paths.

---

Developed by the Department of Radiation Oncology, Nanfang Hospital, Southern Medical University.
