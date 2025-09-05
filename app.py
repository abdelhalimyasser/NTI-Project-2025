import streamlit as st
import joblib
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("Matplotlib is not installed. Confidence score visualization will be skipped.")

# Custom CSS for enhanced styling
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
st.markdown("Predict the likelihood of heart disease using advanced machine learning models. Enter patient details below and get results instantly.")

# Collapsible section for feature explanations
with st.expander("Understanding Input Features"):
    st.markdown("""
    <div class="info-box">
        <strong>Age</strong>: Patient's age in years (0-120).<br>
        <strong>Sex</strong>: 0 = Female, 1 = Male.<br>
        <strong>Chest Pain Type (cp)</strong>: 0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic.<br>
        <strong>Resting Blood Pressure (trestbps)</strong>: Blood pressure in mm Hg at rest (e.g., 90-200).<br>
        <strong>Serum Cholesterol (chol)</strong>: Cholesterol level in mg/dl (e.g., 100-400).<br>
        <strong>Fasting Blood Sugar (fbs)</strong>: 1 if > 120 mg/dl, else 0.<br>
        <strong>Resting ECG Results (restecg)</strong>: 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy.<br>
        <strong>Maximum Heart Rate (thalach)</strong>: Max heart rate during stress test (e.g., 70-200).<br>
        <strong>Exercise Induced Angina (exang)</strong>: 1 = Yes, 0 = No.<br>
        <strong>ST Depression (oldpeak)</strong>: ST depression induced by exercise relative to rest (0-6.2).<br>
        <strong>Slope of ST Segment (slope)</strong>: 0 = Upsloping, 1 = Flat, 2 = Downsloping.<br>
        <strong>Number of Major Vessels (ca)</strong>: Vessels colored by fluoroscopy (0-4).<br>
        <strong>Thalassemia (thal)</strong>: 0 = Not described, 1 = Fixed defect, 2 = Reversible defect, 3 = Normal.
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
        cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], format_func=lambda x: ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"][x], help="Type of chest pain experienced.")

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
        thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3], format_func=lambda x: ["Not described", "Fixed defect", "Reversible defect", "Normal"][x], help="Thalassemia condition.")

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

# Predict when form is submitted
if submit_button:
    # Input validation
    if any(input_data.iloc[0].isna()) or age < 0 or trestbps < 50 or chol < 50 or thalach < 50 or oldpeak < 0:
        st.error("Invalid inputs detected. Please ensure all values are within realistic ranges (e.g., no negative values, blood pressure ≥ 50, cholesterol ≥ 50).")
    else:
        with st.spinner("Analyzing patient data..."):
            # Predictions
            log_pred = logistic_model.predict(input_data)[0]
            log_prob = logistic_model.predict_proba(input_data)[0][1]
            log_prob_no = 1 - log_prob
            
            rf_pred = random_forest_model.predict(input_data)[0]
            rf_prob = random_forest_model.predict_proba(input_data)[0][1]
            rf_prob_no = 1 - rf_prob
            
            # Display predictions
            st.markdown('<div class="subheader">Model Predictions</div>', unsafe_allow_html=True)
            st.write(f"**Logistic Regression**: {'Has Heart Disease' if log_pred == 1 else 'No Heart Disease'} (Confidence: {log_prob if log_pred == 1 else log_prob_no:.2f})")
            st.write(f"**Random Forest**: {'Has Heart Disease' if rf_pred == 1 else 'No Heart Disease'} (Confidence: {rf_prob if rf_pred == 1 else rf_prob_no:.2f})")
            
            # Visualize confidence scores if matplotlib is available
            if MATPLOTLIB_AVAILABLE:
                st.markdown('<div class="subheader">Confidence Scores</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                models = ['Logistic Regression', 'Random Forest']
                confidences = [log_prob if log_pred == 1 else log_prob_no, rf_prob if rf_pred == 1 else rf_prob_no]
                colors = ['#1e88e5' if log_pred == 1 else '#4caf50', '#1e88e5' if rf_pred == 1 else '#4caf50']
                ax.bar(models, confidences, color=colors)
                ax.set_ylim(0, 1)
                ax.set_ylabel("Confidence Score")
                for i, v in enumerate(confidences):
                    ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
                st.pyplot(fig)
            
            # Decide best prediction
            if log_pred == rf_pred:
                best_pred = log_pred
                best_model = "Both Models"
                best_conf = max(log_prob if log_pred == 1 else log_prob_no, rf_prob if rf_pred == 1 else rf_prob_no)
            else:
                if log_prob > rf_prob:
                    best_pred = log_pred
                    best_model = "Logistic Regression"
                    best_conf = log_prob if log_pred == 1 else log_prob_no
                else:
                    best_pred = rf_pred
                    best_model = "Random Forest"
                    best_conf = rf_prob if rf_pred == 1 else rf_prob_no
                
                if best_conf < 0.6:
                    st.warning("Low confidence in prediction (<60%). Results may be unreliable. Consult a medical professional for accurate diagnosis.")
            
            # Final output
            st.markdown('<div class="subheader">Final Prediction</div>', unsafe_allow_html=True)
            if best_pred == 1:
                st.error(f"The patient is predicted to have heart disease based on {best_model} (Confidence: {best_conf:.2f}).")
            else:
                st.success(f"The patient is predicted to NOT have heart disease based on {best_model} (Confidence: {best_conf:.2f}).")
            st.markdown("*Note: This prediction is for informational purposes only. Please consult a healthcare professional for medical advice.*")
