import streamlit as st
import joblib
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("Matplotlib is not installed. Visualization features will be skipped.")
import io
import base64
from datetime import datetime

# Custom CSS for styling
st.markdown("""
<style>
    .main {background-color: #f9f9f9;}
    .stButton>button {
        background-color: #0288d1;
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0277bd;
    }
    .stNumberInput input, .stSelectbox select {
        border-radius: 6px;
        border: 1px solid #ccc;
        padding: 8px;
    }
    .header {font-size: 26px; font-weight: bold; color: #1e88e5; margin-bottom: 15px;}
    .subheader {font-size: 20px; font-weight: bold; color: #424242; margin-top: 20px;}
    .info-box {background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 10px;}
    .stProgress .st-bo {background-color: #4caf50;}
</style>
""", unsafe_allow_html=True)

# Load the pre-trained models
try:
    logistic_model = joblib.load("logistic_model.pkl")
    random_forest_model = joblib.load("random_forest_model.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'logistic_model.pkl' and 'random_forest_model.pkl' are in the same directory.")
    st.stop()

# Streamlit app title
st.title("Heart Disease Prediction App")
st.markdown("Predict the likelihood of heart disease using machine learning models. Enter patient details below to get results.")

# Collapsible section for feature explanations
with st.expander("Understanding Input Features"):
    st.markdown("""
    <div class="info-box">
        <strong>Age</strong>: Patient's age in years (0-120).<br>
        <strong>Sex</strong>: 0 = Female, 1 = Male.<br>
        <strong>Chest Pain Type (cp)</strong>: 0 = Typical angina (less risky), 1 = Atypical angina (high risk), 2 = Non-anginal pain (high risk), 3 = Asymptomatic (moderate risk).<br>
        <strong>Resting Blood Pressure (trestbps)</strong>: Blood pressure in mm Hg at rest (e.g., 90-200).<br>
        <strong>Serum Cholesterol (chol)</strong>: Cholesterol level in mg/dl (e.g., 100-400).<br>
        <strong>Fasting Blood Sugar (fbs)</strong>: 1 if > 120 mg/dl, else 0.<br>
        <strong>Resting ECG Results (restecg)</strong>: 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy.<br>
        <strong>Maximum Heart Rate (thalach)</strong>: Max heart rate during stress test (e.g., 70-200).<br>
        <strong>Exercise Induced Angina (exang)</strong>: 1 = Yes, 0 = No.<br>
        <strong>ST Depression (oldpeak)</strong>: ST depression induced by exercise relative to rest (0-6.2).<br>
        <strong>Slope of ST Segment (slope)</strong>: 0 = Upsloping, 1 = Flat, 2 = Downsloping.<br>
        <strong>Number of Major Vessels (ca)</strong>: Vessels colored by fluoroscopy (0-4).<br>
        <strong>Thalassemia (thal)</strong>: 0 = Not described, 1 = Normal, 2 = Reversible defect (high risk), 3 = Fixed defect (less risky).
    </div>
    """, unsafe_allow_html=True)

# Input form
st.markdown('<div class="header">Patient Details</div>', unsafe_allow_html=True)
with st.form(key="patient_form"):
    st.markdown("**Demographic Information**", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=35, help="Patient's age in years.")
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="Select patient’s gender.")
    with col2:
        cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], format_func=lambda x: ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"][x], help="Type of chest pain. Typical angina (0) is less risky; Atypical (1) and Non-anginal (2) are high risk.")

    st.markdown("**Clinical Measurements**", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=250, value=110, help="Blood pressure in mm Hg at rest.")
        chol = st.number_input("Serum Cholesterol (chol)", min_value=50, max_value=500, value=180, help="Cholesterol level in mg/dl.")
        fbs = st.selectbox("Fasting Blood Sugar (fbs)", options=[0, 1], format_func=lambda x: "≤ 120 mg/dl" if x == 0 else "> 120 mg/dl", help="Fasting blood sugar level.")
    with col4:
        restecg = st.selectbox("Resting ECG Results (restecg)", options=[0, 1, 2], format_func=lambda x: ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"][x], help="ECG results at rest.")
        thalach = st.number_input("Maximum Heart Rate (thalach)", min_value=50, max_value=250, value=180, help="Max heart rate during stress test.")
        exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="Angina induced by exercise.")

    st.markdown("**Advanced Measurements**", unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, help="ST depression relative to rest.")
        slope = st.selectbox("Slope of ST Segment (slope)", options=[0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x], help="Slope of ST segment during exercise.")
    with col6:
        ca = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4], help="Vessels colored by fluoroscopy.")
        thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3], format_func=lambda x: ["Not described", "Normal", "Reversible defect", "Fixed defect"][x], help="Thalassemia condition. Normal (1) or Fixed defect (3) are less risky; Reversible defect (2) is high risk.")

    # Submit button
    submit_button = st.form_submit_button(label="Predict Heart Disease")

# Prepare input data
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Generate PDF report
def generate_pdf_report(input_data, log_pred, log_prob, rf_pred, rf_prob, best_pred, best_model, best_conf):
    latex_content = r"""
    \documentclass{article}
    \usepackage[utf8]{inputenc}
    \usepackage{geometry}
    \geometry{a4paper, margin=1in}
    \usepackage{booktabs}
    \usepackage{xcolor}
    \title{Heart Disease Prediction Report}
    \author{AI-Powered Prediction System}
    \date{\today}
    \begin{document}
    \maketitle
    \section{Patient Details}
    \begin{tabular}{ll}
    \toprule
    \textbf{Feature} & \textbf{Value} \\
    \midrule
    Age & """ + str(input_data['age'][0]) + r""" \\
    Sex & """ + ("Female" if input_data['sex'][0] == 0 else "Male") + r""" \\
    Chest Pain Type & """ + ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"][input_data['cp'][0]] + r""" \\
    Resting Blood Pressure & """ + str(input_data['trestbps'][0]) + r""" mm Hg \\
    Serum Cholesterol & """ + str(input_data['chol'][0]) + r""" mg/dl \\
    Fasting Blood Sugar & """ + ("≤ 120 mg/dl" if input_data['fbs'][0] == 0 else "> 120 mg/dl") + r""" \\
    Resting ECG Results & """ + ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"][input_data['restecg'][0]] + r""" \\
    Maximum Heart Rate & """ + str(input_data['thalach'][0]) + r""" \\
    Exercise Induced Angina & """ + ("No" if input_data['exang'][0] == 0 else "Yes") + r""" \\
    ST Depression & """ + str(input_data['oldpeak'][0]) + r""" \\
    Slope of ST Segment & """ + ["Upsloping", "Flat", "Downsloping"][input_data['slope'][0]] + r""" \\
    Number of Major Vessels & """ + str(input_data['ca'][0]) + r""" \\
    Thalassemia & """ + ["Not described", "Normal", "Reversible defect", "Fixed defect"][input_data['thal'][0]] + r""" \\
    \bottomrule
    \end{tabular}
    \section{Prediction Results}
    \textbf{Logistic Regression}: """ + ("Has Heart Disease" if log_pred == 0 else "No Heart Disease") + r""" (Confidence: """ + f"{log_prob:.2f}" + r""") \\
    \textbf{Random Forest}: """ + ("Has Heart Disease" if rf_pred == 0 else "No Heart Disease") + r""" (Confidence: """ + f"{rf_prob:.2f}" + r""") \\
    \textbf{Final Prediction}: """ + ("The patient is predicted to have heart disease" if best_pred == 0 else "The patient is predicted to NOT have heart disease") + r""" based on """ + best_model + r""" (Confidence: """ + f"{best_conf:.2f}" + r"""). \\
    \section{Disclaimer}
    This report is generated by an AI model for informational purposes only. Consult a healthcare professional for medical advice.
    \end{document}
    """
    return latex_content

# Predict when form is submitted
if submit_button:
    # Input validation
    if any(input_data.iloc[0].isna()) or age < 0 or trestbps < 50 or chol < 50 or thalach < 50 or oldpeak < 0:
        st.error("Invalid inputs detected. Please ensure all values are within realistic ranges (e.g., no negative values, blood pressure ≥ 50, cholesterol ≥ 50).")
    else:
        with st.spinner("Analyzing patient data..."):
            # Predictions
            log_pred = logistic_model.predict(input_data)[0]
            log_prob = logistic_model.predict_proba(input_data)[0][0]  # Probability of class 0 (Has Heart Disease in model)
            
            rf_pred = random_forest_model.predict(input_data)[0]
            rf_prob = random_forest_model.predict_proba(input_data)[0][0]  # Probability of class 0 (Has Heart Disease in model)
            
            # Invert predictions (0 = Has Heart Disease, 1 = No Heart Disease)
            log_pred = 1 - log_pred
            rf_pred = 1 - rf_pred
            log_conf = 1 - log_prob if log_pred == 1 else log_prob
            rf_conf = 1 - rf_prob if rf_pred == 1 else rf_prob
            
            # Display predictions
            st.markdown('<div class="subheader">Model Predictions</div>', unsafe_allow_html=True)
            st.write(f"**Logistic Regression**: {'Has Heart Disease' if log_pred == 0 else 'No Heart Disease'} (Confidence: {log_conf:.2f})")
            st.write(f"**Random Forest**: {'Has Heart Disease' if rf_pred == 0 else 'No Heart Disease'} (Confidence: {rf_conf:.2f})")
            
            # Decide best prediction
            if log_pred == rf_pred:
                best_pred = log_pred
                best_model = "Both Models"
                best_conf = max(log_conf, rf_conf)
            else:
                if log_conf > rf_conf:
                    best_pred = log_pred
                    best_model = "Logistic Regression"
                    best_conf = log_conf
                else:
                    best_pred = rf_pred
                    best_model = "Random Forest"
                    best_conf = rf_conf
                
                if best_conf < 0.6:
                    st.warning("Low confidence in prediction (<60%). Results may be unreliable. Consult a medical professional for accurate diagnosis.")
            
            # Final output
            st.markdown('<div class="subheader">Final Prediction</div>', unsafe_allow_html=True)
            if best_pred == 0:
                st.error(f"The patient is predicted to have heart disease based on {best_model} (Confidence: {best_conf:.2f}).")
            else:
                st.success(f"The patient is predicted to NOT have heart disease based on {best_model} (Confidence: {best_conf:.2f}).")
            
            # Confidence scores bar chart
            if MATPLOTLIB_AVAILABLE:
                st.markdown('<div class="subheader">Confidence Scores</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                models = ['Logistic Regression', 'Random Forest']
                confidences = [log_conf, rf_conf]
                colors = ['#1e88e5' if log_pred == 0 else '#4caf50', '#1e88e5' if rf_pred == 0 else '#4caf50']
                ax.bar(models, confidences, color=colors)
                ax.set_ylim(0, 1)
                ax.set_ylabel("Confidence Score")
                for i, v in enumerate(confidences):
                    ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
                st.pyplot(fig)
            
            # Risk gauge
            st.markdown('<div class="subheader">Risk Level</div>', unsafe_allow_html=True)
            st.progress(min(best_conf, 1.0))
            st.caption(f"Confidence Score: {best_conf:.2f}")
            
            # Feature importance plot
            if MATPLOTLIB_AVAILABLE:
                st.markdown('<div class="subheader">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
                feature_names = input_data.columns
                importances = random_forest_model.feature_importances_
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(feature_names, importances, color='#0288d1')
                ax.set_xlabel("Importance")
                ax.set_title("Feature Importance")
                st.pyplot(fig)
            
            # Downloadable PDF report
            st.markdown('<div class="subheader">Download Report</div>', unsafe_allow_html=True)
            pdf_content = generate_pdf_report(input_data, log_pred, log_conf, rf_pred, rf_conf, best_pred, best_model, best_conf)
            pdf_buffer = io.BytesIO()
            with open("report.tex", "w") as f:
                f.write(pdf_content)
            import subprocess
            try:
                subprocess.run(["latexmk", "-pdf", "report.tex"], check=True)
                with open("report.pdf", "rb") as f:
                    pdf_buffer.write(f.read())
                b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="heart_disease_prediction_report.pdf">Download Prediction Report</a>'
                st.markdown(href, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"PDF generation failed: {e}. Please ensure LaTeX is installed or try again.")
            
            st.markdown("*Note: This prediction is for informational purposes only. Please consult a healthcare professional for medical advice.*")
