# Dynamic Prediction of Chemotherapy-Induced Myelosuppression in NPC

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://myelosuppression-pred.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation, pre-trained models, and clinical application for the study:
**"Dynamic prediction of treatment-related myelosuppression in patients with nasopharyngeal carcinoma using routine longitudinal data: a multicentre real-world study with nested prospective clinical trial validation."**

## 🌟 Clinical Utility & Web Tool
To facilitate clinical translation and real-time monitoring, we have deployed an interactive web-based risk calculator.

👉 **Access the Tool Here: [https://myelosuppression-pred.streamlit.app/](https://myelosuppression-pred.streamlit.app/)**

## 📂 Repository Structure
* `web_tool/`: Source code for the Streamlit-based web application.
* `models/`: Pre-trained models (XGBoost, LightGBM, TabPFN, and Logistic Regression) for Hb, PLT, and WBC/Neutrophil toxicity prediction.
* `notebooks/`:
    * `01_Baseline_Stats.ipynb`: Statistical analysis of the multicentre real-world cohorts.
    * `02_Model_Validation_and_Cutoff.ipynb`: Performance evaluation and safety-first thresholding.
    * `03_SHAP_Explanation.ipynb`: Global and local model interpretability (SHAP).
* `data/`: Contains synthetic demo data for pipeline testing. (Note: Real patient data is not public due to privacy regulations).

## 🚀 Research Background
* **Cohort Size:** Developed using a large-scale real-world dataset of over 12,000 patients from multiple centers (Nanfang Hospital, SYSUCC, etc.).
* **Prospective Validation:** Validated across five centers using data from three nested prospective clinical trials (**NCT03919552, NCT06767488, NCT06017895**).
* **Framework:** Incorporates routine laboratory data during radiotherapy to provide dynamic, real-time risk stratification.

## 🛠️ Getting Started

### 1. Installation
```bash
git clone [https://github.com/TingxiTang/NPC-Myelosuppression-Dynamic-Prediction.git](https://github.com/TingxiTang/NPC-Myelosuppression-Dynamic-Prediction.git)
cd NPC-Myelosuppression-Dynamic-Prediction
pip install -r requirements.txt
```

### 2. Run Analysis
Open the provided Jupyter notebooks in the notebooks/ directory to reproduce validation metrics and SHAP visualizations.

### 3.Run Web Tool Locally
```bash
streamlit run web_tool/app.py
```

## 🔐 Data Availability & Transparency
The raw clinical datasets from Nanfang Hospital and other participating centers are not publicly available due to patient confidentiality and institutional data security policies. However, de-identified data may be available from the corresponding author (Jian Guan, Guanj@smu.edu.cn) upon reasonable request and institutional approval.

---

Developed by the Department of Radiation Oncology, Nanfang Hospital, Southern Medical University.
