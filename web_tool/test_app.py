#!/usr/bin/env python3
"""
Test script to test the Streamlit app with test CSV files and generate PDFs
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
import shap
import io
import traceback
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# Import the functions from app.py
from app import prepare_input_vector, create_pdf_report, load_resources, process_drug_features

def test_prediction(csv_file, output_dir, language="Chinese"):
    """Test prediction with a CSV file and generate PDF report"""
    print(f"Testing {csv_file} with language {language}")

    # Load resources
    drug_map, feat_names, scaler, scale_cols, mlb_d, mlb_c, models = load_resources()
    if drug_map is None:
        print("Failed to load resources")
        return False

    # Load test data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows from {csv_file}")

    # Test drugs: Cisplatin and Docetaxel
    selected_drugs = ["Cisplatin", "Docetaxel"]

    # Manual features (same as UI defaults)
    manual_feats = {
        "gender": 0,  # Male
        "c_t_stage": 3,
        "c_n_stage": 3,
        "c_m_stage": 0,
        "clinic_stage": 4,
        "ABO": "A"
    }

    # Prepare input vector
    X_input, original_values = prepare_input_vector(
        df, selected_drugs, drug_map, feat_names, scaler, scale_cols, mlb_d, mlb_c, manual_feats
    )

    # Test all three models
    for model_name in ["Hb", "PLT", "WBC_Neut"]:
        print(f"  Testing {model_name} model...")
        model = models[model_name]
        probs = model.predict_proba(X_input)
        prob = probs[0][1]

        # Cutoff values
        cutoffs = {
            "Hb": 0.0076,
            "PLT": 0.0093,
            "WBC_Neut": 0.0039
        }
        cutoff = cutoffs[model_name]
        risk_class = 1 if prob >= cutoff else 0

        # SHAP explanation
        if hasattr(model, 'base_model'):
            core_model = model.base_model
        else:
            core_model = model

        explainer = shap.TreeExplainer(core_model)
        shap_values = explainer.shap_values(X_input)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
            ev = explainer.expected_value[1]
        else:
            sv = shap_values[0]
            ev = explainer.expected_value

        # Create original values series for SHAP
        original_values_list = []
        for col in feat_names:
            val = original_values.get(col, 0.0)
            try:
                val = float(val)
            except:
                val = 0.0
            original_values_list.append(val)
        original_values_series = pd.Series(original_values_list, index=feat_names)

        # Generate force plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 4))
        shap.force_plot(
            ev, sv, original_values_series,
            matplotlib=True, show=False, text_rotation=45
        )
        force_plot_buf = io.BytesIO()
        plt.savefig(force_plot_buf, format='png', bbox_inches='tight', dpi=150)
        force_plot_buf.seek(0)
        plt.close()

        # Generate PDF report
        pdf_bytes = create_pdf_report(
            df.iloc[0].to_dict(),
            selected_drugs,
            model_name,
            prob,
            risk_class,
            shap_plot_buf=force_plot_buf,
            language=language
        )

        # Save PDF
        csv_stem = Path(csv_file).stem
        pdf_filename = f"{csv_stem}_{model_name}_{language}.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)
        with open(pdf_path, 'wb') as f:
            f.write(pdf_bytes)
        print(f"    Generated {pdf_path}")

    return True

def main():
    """Main test function"""
    test_dir = Path("test")
    output_dir = test_dir  # Save PDFs in the same test directory

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Test files
    test_files = [
        test_dir / "test_low_risk.csv",
        test_dir / "test_high_risk.csv"
    ]

    # Test both languages
    languages = ["Chinese", "English"]

    for test_file in test_files:
        if test_file.exists():
            for lang in languages:
                try:
                    success = test_prediction(str(test_file), str(output_dir), lang)
                    if success:
                        print(f"✓ Successfully tested {test_file.name} with {lang} language")
                    else:
                        print(f"✗ Failed to test {test_file.name} with {lang} language")
                except Exception as e:
                    print(f"✗ Error testing {test_file.name} with {lang} language: {e}")
                    traceback.print_exc()
        else:
            print(f"✗ Test file {test_file} does not exist")

    print("Test completed!")

if __name__ == "__main__":
    main()