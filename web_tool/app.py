import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import traceback
import io
import datetime
import os
import sys
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from pathlib import Path

# ==========================================
# 1. 核心类定义 (必须在 pickle 加载前定义)
# ==========================================

class PlattScalingCalibrator:
    """
    手动实现 Platt Scaling 概率校准
    """
    def __init__(self, base_model):
        self.base_model = base_model
        self.platt_lr = None

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        """校准后的概率预测"""
        if hasattr(self.base_model, "predict_proba"):
            raw_probs = self.base_model.predict_proba(X)[:, 1]
        else:
            raw_probs = self.base_model.predict(X)

        if self.platt_lr is not None:
            # 如果已有校准器，使用它
            calibrated_probs = self.platt_lr.predict_proba(raw_probs.reshape(-1, 1))
            return calibrated_probs
        else:
            # 如果没有校准器（异常情况），手动构建 (N, 2) 格式返回
            return np.column_stack((1 - raw_probs, raw_probs))

# 注入到 sys.modules 确保 pickle 可以找到该类
sys.modules['__main__'].PlattScalingCalibrator = PlattScalingCalibrator


# ==========================================
# 2. 配置与常量
# ==========================================

st.set_page_config(
    page_title="骨髓抑制风险预测系统",
    page_icon="🏥",
    layout="wide"
)

# 你的工作目录
WORKING_DIR = Path(__file__).parent.absolute()

# macOS 中文字体路径 (用于解决 PDF 乱码)
FONT_PATH = "/System/Library/Fonts/PingFang.ttc"
if not os.path.exists(FONT_PATH):
    # 备用路径
    FONT_PATH = "/System/Library/Fonts/STHeiti Light.ttc"

# ==========================================
# 3. 辅助函数：PDF 生成 (解决乱码)
# ==========================================

def register_chinese_font():
    """注册中文字体"""
    try:
        if os.path.exists(FONT_PATH):
            pdfmetrics.registerFont(TTFont('ChineseFont', FONT_PATH))
            return True
        return False
    except Exception as e:
        print(f"字体注册失败: {e}")
        return False

def create_pdf_report(patient_data, selected_drugs, model_name, prob, risk_class, shap_plot_buf=None, language="Chinese"):
    """生成 PDF 报告"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)

    # 注册字体
    has_font = register_chinese_font()
    font_name = 'ChineseFont' if has_font else 'Helvetica'

    styles = getSampleStyleSheet()

    # 自定义样式
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontName=font_name,
        fontSize=24,
        leading=30,
        alignment=1 # Center
    )

    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading2'],
        fontName=font_name,
        fontSize=14,
        leading=20,
        textColor=colors.darkblue
    )

    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        leading=14
    )

    story = []

    # 1. 标题
    if language == "Chinese":
        title_text = "骨髓抑制风险预测报告"
    else:
        title_text = "Bone Marrow Suppression Risk Prediction Report"
    story.append(Paragraph(title_text, title_style))
    story.append(Spacer(1, 20))

    # 2. 预测结果摘要
    if language == "Chinese":
        summary_text = "预测结果摘要"
        model_label = "预测模型"
        score_label = "风险评分"
        class_label = "风险类别"
        high_risk_text = "高风险 (High Risk)"
        low_risk_text = "低风险 (Low Risk)"
    else:
        summary_text = "Prediction Summary"
        model_label = "Prediction Model"
        score_label = "Risk Score"
        class_label = "Risk Category"
        high_risk_text = "High Risk"
        low_risk_text = "Low Risk"

    story.append(Paragraph(summary_text, header_style))
    risk_text = high_risk_text if risk_class == 1 else low_risk_text

    result_data = [
        [model_label, model_name],
        [score_label, f"{prob:.4f}"],
        [class_label, risk_text]
    ]

    t_result = Table(result_data, colWidths=[150, 300])
    t_result.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('Padding', (0, 0), (-1, -1), 12),
    ]))
    story.append(t_result)
    story.append(Spacer(1, 20))

    # 3. 患者信息
    if language == "Chinese":
        patient_info_text = "患者基本信息"
        gender_label = "性别"
        age_label = "年龄"
        stage_label = "临床分期"
        drug_label = "药物方案"
        no_drugs_text = "无"
        male_text = "男"
        female_text = "女"
    else:
        patient_info_text = "Patient Basic Information"
        gender_label = "Gender"
        age_label = "Age"
        stage_label = "Clinical Stage"
        drug_label = "Treatment Plan"
        no_drugs_text = "None"
        male_text = "Male"
        female_text = "Female"

    story.append(Paragraph(patient_info_text, header_style))
    p_info = [
        [gender_label, female_text if patient_data.get('gender') == 1 else male_text],
        [age_label, str(patient_data.get('age', 'N/A'))],
        [stage_label, str(patient_data.get('clinic_stage', 'N/A'))],
        [drug_label, ", ".join(selected_drugs) if selected_drugs else no_drugs_text]
    ]
    t_info = Table(p_info, colWidths=[150, 300])
    t_info.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
    ]))
    story.append(t_info)
    story.append(Spacer(1, 20))

    # 4. SHAP 分析图 (Force Plot)
    if language == "Chinese":
        shap_text = "特征重要性分析 (SHAP Force Plot)"
        no_shap_text = "未生成 SHAP 图像"
    else:
        shap_text = "Feature Importance Analysis (SHAP Force Plot)"
        no_shap_text = "SHAP plot not generated"

    story.append(Paragraph(shap_text, header_style))
    if shap_plot_buf:
        shap_plot_buf.seek(0)
        # Force Plot 通常比较宽，设置合适的宽高
        img = ReportLabImage(shap_plot_buf, width=540, height=200)
        story.append(img)
    else:
        story.append(Paragraph(no_shap_text, normal_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ==========================================
# 4. 核心逻辑：数据加载与处理
# ==========================================

@st.cache_resource
def load_resources():
    """加载所有资源：映射表、Scaler、模型等"""
    try:
        # 1. 药物映射 - 保留 <RT>，去除 PAD/UNK
        drug_df = pd.read_csv(f"{WORKING_DIR}/drug_category_index_clean.csv")
        # 使用英文药物名称作为标准（模型训练时使用英文名）
        drug_mapping = drug_df[~drug_df['Drug_Name(En)'].isin(['<PAD>', '<UNK>'])]

        # 2. 特征名称
        feature_names = joblib.load(f"{WORKING_DIR}/lightgbm_feature_names.joblib")

        # 3. 标准化器 (Scaler)
        # scaler 用于标准化连续变量
        # scaler_continuous_features.joblib 是特征名列表
        scaler = joblib.load(f"{WORKING_DIR}/scaler_continuous.joblib")
        scale_cols = joblib.load(f"{WORKING_DIR}/scaler_continuous_features.joblib")

        # 4. MultiLabelBinarizer
        mlb_drug = joblib.load(f"{WORKING_DIR}/mlb_drug.joblib")
        mlb_category = joblib.load(f"{WORKING_DIR}/mlb_category.joblib")

        # 5. 模型
        models = {
            "Hb": joblib.load(f"{WORKING_DIR}/Hb_LightGBM_Calibrated.joblib"),
            "PLT": joblib.load(f"{WORKING_DIR}/PLT_LightGBM_Calibrated.joblib"),
            "WBC_Neut": joblib.load(f"{WORKING_DIR}/WBC_Neut_LightGBM_Calibrated.joblib")
        }

        return drug_mapping, feature_names, scaler, scale_cols, mlb_drug, mlb_category, models
    except Exception as e:
        st.error(f"资源加载失败: {str(e)}")
        st.error(traceback.format_exc())
        return None, None, None, None, None, None, None

def process_drug_features(selected_drugs, drug_mapping, mlb_drug, mlb_category):
    """
    处理药物特征：
    1. 映射 ID 并进行多热编码
    2. 计算当前药物方案的 is_immuno, is_target, is_rt, is_chemo
    注意：cum_ 系列特征由用户上传，不在此计算
    """
    drug_ids = []
    category_ids = []

    # 统计量初始化 (仅用于计算 is_ 标志位)
    current_stats = {
        'is_immuno': 0, 'is_target': 0, 'is_rt': 0, 'is_chemo': 0
    }

    if not selected_drugs:
        drug_encoded = mlb_drug.transform([[]])
        category_encoded = mlb_category.transform([[]])
        return drug_encoded, category_encoded, current_stats

    # 临时的计数器，用于判断是否 > 0
    count_immuno = 0
    count_target = 0
    count_rt = 0
    count_chemo = 0

    for drug in selected_drugs:
        row = drug_mapping[drug_mapping['Drug_Name(En)'] == drug]
        if not row.empty:
            d_id = int(row['Drug_ID'].iloc[0])
            c_id = int(row['Category_ID'].iloc[0])
            c_name = row['Category_Name'].iloc[0]

            drug_ids.append(d_id)
            category_ids.append(c_id)

            # 逻辑：取 '_' 前缀
            # 特殊情况：RT 没有下划线，直接等于 RT
            prefix = c_name.split('_')[0]

            if prefix == 'Immuno': count_immuno += 1
            if prefix == 'Target': count_target += 1
            if prefix == 'RT' or c_name == 'RT': count_rt += 1
            if prefix == 'Chemo':  count_chemo += 1

    # 生成标志位
    current_stats['is_immuno'] = 1 if count_immuno > 0 else 0
    current_stats['is_target'] = 1 if count_target > 0 else 0
    current_stats['is_rt'] = 1 if count_rt > 0 else 0
    current_stats['is_chemo'] = 1 if count_chemo > 0 else 0

    # 多热编码
    drug_encoded = mlb_drug.transform([drug_ids])
    category_encoded = mlb_category.transform([category_ids])

    return drug_encoded, category_encoded, current_stats

def prepare_input_vector(df, selected_drugs, drug_mapping, feature_names, scaler, scale_cols, mlb_drug, mlb_category, manual_features):
    """
    准备输入向量
    返回: (standardized_df, original_values_dict)
    """
    # 1. 基础数据复制 (来自上传的 CSV)
    # 确保包含了 cum_chemo, cum_rt, cum_target, cum_immuno, is_first_cycle, age 等
    data = df.iloc[0].to_dict()
    original_values = data.copy()  # 保存原始值用于SHAP显示

    # 2. 覆盖/添加手动选择的特征 (gender, stages, ABO)
    data.update(manual_features)
    original_values.update(manual_features)

    # ABO 独热编码 (ABO_A, ABO_B ...)
    abo = data.get('ABO', 'A') # 这里的 ABO 来自 manual_features
    for t in ['A', 'B', 'O', 'AB', '未查']:
        data[f'ABO_{t}'] = 1 if abo == t else 0
        original_values[f'ABO_{t}'] = 1 if abo == t else 0

    # 3. 药物处理 (获取 embedding 和当次药物属性)
    drug_enc, cat_enc, drug_stats = process_drug_features(selected_drugs, drug_mapping, mlb_drug, mlb_category)

    # 将 is_immuno 等更新到 data 中 (覆盖 CSV 中可能存在的空值或旧值)
    data.update(drug_stats)
    original_values.update(drug_stats)

    # 4. 标准化处理
    if scaler is not None and scale_cols is not None:
        try:
            # 构建待标准化的 DataFrame
            # 注意：必须严格按照 fit 时的特征顺序 (即 scale_cols 中的顺序)
            vals_to_scale = {}
            for col in scale_cols:
                # 转 float
                v = data.get(col, 0.0)
                try: v = float(v)
                except: v = 0.0
                vals_to_scale[col] = [v]

            df_scale = pd.DataFrame(vals_to_scale, columns=scale_cols)
            scaled_vals_matrix = scaler.transform(df_scale)

            # 将标准化后的值更新回 data
            # scaled_vals_matrix shape (1, N)
            for i, col in enumerate(scale_cols):
                data[col] = scaled_vals_matrix[0][i]

        except Exception as e:
            st.warning(f"标准化过程出现警告: {e}")
            # 继续执行，使用原始值

    # 5. 构建数值部分向量 (LightGBM 特征顺序)
    # 需要严格按照 feature_names 的顺序构建
    n_drug_feats = len(mlb_drug.classes_)
    n_cat_feats = len(mlb_category.classes_)

    # 数值特征是 feature_names 去掉后部的 drug 和 category
    numeric_feat_names = feature_names[:-(n_drug_feats + n_cat_feats)]

    num_vals = []
    for col in numeric_feat_names:
        # 从 data 中获取 (此时已经是标准化后的值)
        val = data.get(col, 0.0)
        try:
            val = float(val)
        except:
            val = 0.0
        num_vals.append(val)

    numeric_array = np.array(num_vals).reshape(1, -1)

    # 6. 最终拼接
    final_vector = np.hstack([numeric_array, drug_enc, cat_enc])

    # 生成带列名的 DataFrame (LightGBM 需要列名匹配)
    final_df = pd.DataFrame(final_vector, columns=feature_names)

    return final_df, original_values

def generate_test_csv():
    """
    生成包含所有必需列的测试 CSV (基于 Web_test.csv 的真实样本)
    """
    # 来自 debug 分析的真实第2行数据结构
    # 包含 cum_* 和 is_first_cycle
    test_data = {
        'age': 27.0,
        'is_drinking': 1,
        'is_smoking': 1,
        'base_ALB': 46.4,
        'base_ALB/GLO': 1.6,
        'base_ALT': 83.3,
        'base_AST': 46.1,
        'base_AST/ALT': 0.6,
        'base_Baso': 0.0,
        'base_Baso%': 0.3,
        'base_Ca': 2.21,
        'base_Cl-': 100.2,
        'base_Crea': 67.0,
        'base_DBIL': 6.2,
        'base_Eos': 0.0,
        'base_Eos%': 0.1,
        'base_GLO': 29.8,
        'base_Hb': 157.0,
        'base_Hct': 45.4,
        'base_IBIL': 11.2,
        'base_K+': 3.9,
        'base_Lymph': 1.4,
        'base_Lymph%': 19.3,
        'base_MCH': 29.4,
        'base_MCHC': 346.0,
        'base_MCV': 85.0,
        'base_MPV': 10.6,
        'base_Mg2+': 0.86,
        'base_Mono': 0.4,
        'base_Mono%': 0.4, # 修正
        'base_Na+': 137.9,
        'base_Neut': 5.6,
        'base_Neut%': 75.2,
        'base_P': 1.15,
        'base_P-LCR%': 30.1,
        'base_PCT': 0.28, # 修正
        'base_PDW': 12.3,
        'base_PLT': 266.0,
        'base_RBC': 5.34,
        'base_RDW-CV': 12.6,
        'base_RDW-SD': 38.6,
        'base_TBIL': 17.4,
        'base_TP': 76.2,
        'base_UA': 365.0,
        'base_Urea': 4.1,
        'base_WBC': 7.4,
        # prev_nadir cols (简化为0或合理值)
        'prev_nadir_ALB': 0,'prev_nadir_ALB/GLO':0,'prev_nadir_ALT':0,'prev_nadir_AST':0,
        'prev_nadir_AST/ALT':0,'prev_nadir_Baso':0,'prev_nadir_Baso%':0,'prev_nadir_Ca':0,
        'prev_nadir_Cl-':0,'prev_nadir_Crea':0,'prev_nadir_DBIL':0,'prev_nadir_Eos':0,
        'prev_nadir_Eos%':0,'prev_nadir_GLO':0,'prev_nadir_Hb':0,'prev_nadir_Hct':0,
        'prev_nadir_IBIL':0,'prev_nadir_K+':0,'prev_nadir_Lymph':0,'prev_nadir_Lymph%':0,
        'prev_nadir_MCH':0,'prev_nadir_MCHC':0,'prev_nadir_MCV':0,'prev_nadir_MPV':0,
        'prev_nadir_Mg2+':0,'prev_nadir_Mono':0,'prev_nadir_Mono%':0,'prev_nadir_Na+':0,
        'prev_nadir_Neut':0,'prev_nadir_Neut%':0,'prev_nadir_P':0,'prev_nadir_P-LCR%':0,
        'prev_nadir_PCT':0,'prev_nadir_PDW':0,'prev_nadir_PLT':0,'prev_nadir_RBC':0,
        'prev_nadir_RDW-CV':0,'prev_nadir_RDW-SD':0,'prev_nadir_TBIL':0,'prev_nadir_TP':0,
        'prev_nadir_UA':0,'prev_nadir_Urea':0,'prev_nadir_WBC':0,
        # 关键的 cum_ 特征
        'cum_chemo': 1.0,
        'cum_target': 1.0,
        'cum_rt': 1.0,
        'cum_immuno': 0.0,
        'is_first_cycle': 1
    }

    df = pd.DataFrame([test_data])
    return df.to_csv(index=False).encode('utf-8')

# ==========================================
# 5. 主应用逻辑 (Main)
# ==========================================

def main():
    # 状态初始化
    if "data" not in st.session_state: st.session_state.data = None
    if "result" not in st.session_state: st.session_state.result = None
    if "language" not in st.session_state: st.session_state.language = "Chinese"

    # 语言切换
    lang_col1, lang_col2, lang_col3 = st.columns([1, 1, 4])
    with lang_col1:
        if st.button("🇨🇳 中文"):
            st.session_state.language = "Chinese"
            st.rerun()
    with lang_col2:
        if st.button("🇺🇸 English"):
            st.session_state.language = "English"
            st.rerun()

    # 根据语言设置标题
    if st.session_state.language == "Chinese":
        st.title("🏥 骨髓抑制风险预测系统")
    else:
        st.title("🏥 Bone Marrow Suppression Risk Prediction System")

    # 1. 资源加载
    drug_map, feat_names, scaler, scale_cols, mlb_d, mlb_c, models = load_resources()

    if drug_map is None:
        st.stop()

    # 2. 侧边栏：生成测试数据
    with st.sidebar:
        if st.session_state.language == "Chinese":
            st.header("🛠 测试工具")
            if st.button("生成测试患者CSV"):
                csv_data = generate_test_csv()
                st.download_button("📥 下载 test_patient.csv", csv_data, "test_patient.csv", "text/csv")
                st.info("已生成包含 cum_chemo 等必要列的真实测试数据")
        else:
            st.header("🛠 Test Tools")
            if st.button("Generate Test Patient CSV"):
                csv_data = generate_test_csv()
                st.download_button("📥 Download test_patient.csv", csv_data, "test_patient.csv", "text/csv")
                st.info("Generated real test data containing required columns like cum_chemo")

    # 3. 步骤一：上传数据
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.language == "Chinese":
            st.subheader("1. 上传患者指标 (CSV)")
            uploaded = st.file_uploader("选择CSV文件", type="csv")
            if uploaded:
                st.session_state.data = pd.read_csv(uploaded)
                st.success(f"已加载: {len(st.session_state.data)} 行数据")

                # 简单检查列是否存在 (可选)
                req_cols = ['cum_chemo', 'cum_rt', 'is_first_cycle']
                missing = [c for c in req_cols if c not in st.session_state.data.columns]
                if missing:
                    st.warning(f"警告: 上传文件缺少以下列，可能导致预测不准: {missing}")
        else:
            st.subheader("1. Upload Patient Metrics (CSV)")
            uploaded = st.file_uploader("Select CSV file", type="csv")
            if uploaded:
                st.session_state.data = pd.read_csv(uploaded)
                st.success(f"Loaded: {len(st.session_state.data)} rows of data")

                # Simple column check (optional)
                req_cols = ['cum_chemo', 'cum_rt', 'is_first_cycle']
                missing = [c for c in req_cols if c not in st.session_state.data.columns]
                if missing:
                    st.warning(f"Warning: Uploaded file is missing the following columns, which may affect prediction accuracy: {missing}")

    # 4. 步骤二：UI 输入特征
    with col2:
        if st.session_state.language == "Chinese":
            st.subheader("2. 补充临床信息")
            gender = st.selectbox("性别", ["男", "女"], index=0) # 0=男
            gender_val = 0 if gender == "男" else 1

            c_t = st.selectbox("cT 分期", [0, 1, 2, 3, 4], index=3)
            c_n = st.selectbox("cN 分期", [0, 1, 2, 3], index=3)
            c_m = st.selectbox("cM 分期", [0, 1], index=0)

            # 简单计算 clinic_stage (示例: 1-4)
            clinic_stage = min(4, max(1, c_t + c_n))
            st.caption(f"自动计算临床分期: {clinic_stage}")

            abo = st.selectbox("ABO 血型", ["A", "B", "O", "AB", "未查"], index=0)
        else:
            st.subheader("2. Supplement Clinical Information")
            gender = st.selectbox("Gender", ["Male", "Female"], index=0) # 0=Male
            gender_val = 0 if gender == "Male" else 1

            c_t = st.selectbox("cT Stage", [0, 1, 2, 3, 4], index=3)
            c_n = st.selectbox("cN Stage", [0, 1, 2, 3], index=3)
            c_m = st.selectbox("cM Stage", [0, 1], index=0)

            # Simple clinic_stage calculation (example: 1-4)
            clinic_stage = min(4, max(1, c_t + c_n))
            st.caption(f"Automatically calculated clinical stage: {clinic_stage}")

            abo = st.selectbox("ABO Blood Type", ["A", "B", "O", "AB", "Not Checked"], index=0)

    # 5. 步骤三：药物选择
    if st.session_state.language == "Chinese":
        st.subheader("3. 治疗方案")
        # 默认选中 顺铂 和 多西他赛
        default_drugs = ["Cisplatin", "Docetaxel"]
    else:
        st.subheader("3. Treatment Plan")
        # Default selected: Cisplatin and Docetaxel
        default_drugs = ["Cisplatin", "Docetaxel"]

    # 根据语言选择显示的药物名称列
    if st.session_state.language == "Chinese":
        drug_display_names = sorted(drug_map['Drug_Name(Cn)'].dropna().unique())
        drug_en_names = sorted(drug_map['Drug_Name(En)'].dropna().unique())
        # 创建中英文映射
        drug_cn_to_en = dict(zip(drug_map['Drug_Name(Cn)'], drug_map['Drug_Name(En)']))
        drug_en_to_cn = dict(zip(drug_map['Drug_Name(En)'], drug_map['Drug_Name(Cn)']))
    else:
        drug_display_names = sorted(drug_map['Drug_Name(En)'].dropna().unique())
        drug_en_names = drug_display_names
        drug_cn_to_en = {}
        drug_en_to_cn = {}

    # 默认选中 顺铂 和 多西他赛
    pre_select_display = []
    for drug in default_drugs:
        if st.session_state.language == "Chinese":
            # 找到对应的中文名
            cn_name = drug_en_to_cn.get(drug, drug)
            if cn_name in drug_display_names:
                pre_select_display.append(cn_name)
        else:
            if drug in drug_display_names:
                pre_select_display.append(drug)

    selected_drugs_display = st.multiselect(
        "选择化疗药物 (包含 <RT>)" if st.session_state.language == "Chinese" else "Select chemotherapy drugs (including <RT>)",
        drug_display_names,
        default=pre_select_display
    )

    # 将显示名称转换回英文名用于模型处理
    if st.session_state.language == "Chinese":
        selected_drugs = [drug_cn_to_en.get(d, d) for d in selected_drugs_display]
    else:
        selected_drugs = selected_drugs_display

    # 6. 预测与结果
    st.divider()
    if st.session_state.language == "Chinese":
        model_choice = st.radio("选择预测模型", ["Hb", "PLT", "WBC_Neut"], horizontal=True)
        analyze_button_text = "🚀 开始分析"
    else:
        model_choice = st.radio("Select Prediction Model", ["Hb", "PLT", "WBC_Neut"], horizontal=True)
        analyze_button_text = "🚀 Start Analysis"

    if st.button(analyze_button_text, type="primary"):
        if st.session_state.data is None:
            if st.session_state.language == "Chinese":
                st.error("请先上传CSV文件")
            else:
                st.error("Please upload a CSV file first")
            return

        try:
            # 准备输入
            manual_feats = {
                "gender": gender_val,
                "c_t_stage": c_t,
                "c_n_stage": c_n,
                "c_m_stage": c_m,
                "clinic_stage": clinic_stage,
                "ABO": abo
            }

            X_input, original_values = prepare_input_vector(
                st.session_state.data,
                selected_drugs,
                drug_map,
                feat_names,
                scaler,
                scale_cols,
                mlb_d,
                mlb_c,
                manual_feats
            )

            # 模型预测
            model = models[model_choice]
            # Predict Proba
            probs = model.predict_proba(X_input)
            prob = probs[0][1] # 取正类概率

            # 定义 Cutoff 阈值
            cutoffs = {
                "Hb": 0.0076,
                "PLT": 0.0093,
                "WBC_Neut": 0.0039
            }
            cutoff = cutoffs.get(model_choice, 0.5)

            risk_class = 1 if prob >= cutoff else 0

            # SHAP 解释 (Force Plot)
            # 注意：使用 base_model 进行解释
            if hasattr(model, 'base_model'):
                core_model = model.base_model
            else:
                core_model = model

            explainer = shap.TreeExplainer(core_model)
            shap_values = explainer.shap_values(X_input)

            # 统一处理 SHAP 输出
            if isinstance(shap_values, list):
                # Binary classification produces list [negative, positive]
                # LightGBM 通常返回 [array(N,M), array(N,M)]
                sv = shap_values[1][0] # 正类, 第0个样本 -> vector
                ev = explainer.expected_value[1]
            else:
                sv = shap_values[0] # TreeExplainer 有时直接返回 matrix
                ev = explainer.expected_value

            # 保存结果到 Session
            st.session_state.result = {
                "prob": prob,
                "class": risk_class,
                "sv": sv,
                "ev": ev,
                "input_df": X_input,
                "original_values": original_values,
                "model": model_choice,
                "cutoff": cutoff
            }

        except Exception as e:
            st.error(f"预测过程出错: {str(e)}")
            st.error(traceback.format_exc())

    # 7. 结果展示区
    if st.session_state.result:
        res = st.session_state.result

        if st.session_state.language == "Chinese":
            st.subheader("📊 分析结果")
            risk_text = "高风险" if res['class']==1 else "低风险"
            safety_text = "-安全" if res['class']==0 else "+注意"
            model_text = f"模型: {res['model']}"
            shap_title = "特征影响分析 (Force Plot)"
            pdf_button_text = "📄 下载完整报告 (PDF)"
        else:
            st.subheader("📊 Analysis Results")
            risk_text = "High Risk" if res['class']==1 else "Low Risk"
            safety_text = "-Safe" if res['class']==0 else "+Caution"
            model_text = f"Model: {res['model']}"
            shap_title = "Feature Impact Analysis (Force Plot)"
            pdf_button_text = "📄 Download Full Report (PDF)"

        c1, c2, c3 = st.columns(3)
        if st.session_state.language == "Chinese":
            c1.metric("风险评分", f"{res['prob']:.4f}")
            c2.metric("风险类别", f"{risk_text} (CutOff: {res['cutoff']:.3f})", delta=safety_text, delta_color="inverse")
        else:
            c1.metric("Risk Score", f"{res['prob']:.4f}")
            c2.metric("Risk Category", f"{risk_text} (CutOff: {res['cutoff']:.3f})", delta=safety_text, delta_color="inverse")
        c3.info(model_text)

        # SHAP Force Plot
        st.subheader(shap_title)

        # 生成 Force Plot
        try:
            # 创建原始值的DataFrame用于SHAP显示
            original_values = res['original_values']
            feature_names = res['input_df'].columns.tolist()
            original_values_list = []
            for col in feature_names:
                val = original_values.get(col, 0.0)
                try:
                    val = float(val)
                except:
                    val = 0.0
                original_values_list.append(val)
            original_values_series = pd.Series(original_values_list, index=feature_names)

            plt.figure(figsize=(20, 4))
            # matplotlib=True 会直接在当前 figure 上画图
            shap.force_plot(
                res['ev'],
                res['sv'],
                original_values_series,
                matplotlib=True,
                show=False,
                text_rotation=45
            )

            # 保存到 Buffer
            force_plot_buf = io.BytesIO()
            plt.savefig(force_plot_buf, format='png', bbox_inches='tight', dpi=150)
            force_plot_buf.seek(0)

            # 显示
            st.image(force_plot_buf, caption="SHAP Force Plot", use_column_width=True)

            # PDF 下载
            pdf_bytes = create_pdf_report(
                st.session_state.data.iloc[0].to_dict(), # 简化的患者信息
                selected_drugs,
                res['model'],
                res['prob'],
                res['class'],
                shap_plot_buf=force_plot_buf, # 传入 buffer
                language=st.session_state.language
            )

            st.download_button(
                pdf_button_text,
                pdf_bytes,
                f"Risk_Report_{res['model']}.pdf",
                "application/pdf",
                type="primary"
            )
        except Exception as e:
            st.error(f"绘图失败: {e}")

if __name__ == "__main__":
    main()
