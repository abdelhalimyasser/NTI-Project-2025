import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Custom CSS for better styling
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stNumberInput input {
        border-radius: 5px;
    }
    .stSelectbox select {
        border-radius: 5px;
    }
    .header {font-size: 24px; font-weight: bold; margin-bottom: 10px;}
    .subheader {font-size: 18px; font-weight: bold; margin-top: 20px;}
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
st.markdown("Enter patient details below to predict the likelihood of heart disease using machine learning models.")

# Main content for user inputs
st.markdown('<div class="header">Patient Details</div>', unsafe_allow_html=True)

# Create a form for better organization
with st.form(key="patient_form"):
    st.markdown("**Demographic Information**", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=50, help="Patient's age in years (0-120).")
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="0 = Female, 1 = Male")
    with col2:
        cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")

    st.markdown("**Clinical Measurements**", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=0, max_value=300, value=120, help="Resting blood pressure in mm Hg (e.g., 120-200).")
        chol = st.number_input("Serum Cholesterol (chol)", min_value=0, max_value=600, value=200, help="Serum cholesterol in mg/dl (e.g., 100-400).")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], help="1 if fasting blood sugar > 120 mg/dl, else 0.")
    with col4:
        restecg = st.selectbox("Resting ECG Results (restecg)", options=[0, 1, 2], help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy.")
        thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=0, max_value=250, value=150, help="Max heart rate during stress test (e.g., 70-200).")
        exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], help="1 if exercise-induced angina, else 0.")

    st.markdown("**Advanced Measurements**", unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="ST depression relative to rest (0-6.2).")
        slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", options=[0, 1, 2], help="0: Upsloping, 1: Flat, 2: Downsloping.")
    with col6:
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", options=[0, 1, 2, 3, 4], help="Number of major vessels (0-4).")
        thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3], help="0: Normal, 1: Fixed defect, 2: Reversible defect, 3: Not described.")

    # Submit button for the form
    submit_button = st.form_submit_button(label="Predict Heart Disease")

# Prepare input data as DataFrame
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

# Predict when the form is submitted
if submit_button:
    # Input validation
    if any(input_data.iloc[0].isna()) or age < 0 or trestbps < 0 or chol < 0 or thalach < 0 or oldpeak < 0:
        st.error("Please ensure all inputs are valid and non-negative.")
    else:
        with st.spinner("Making predictions..."):
            # Predictions
            log_pred = logistic_model.predict(input_data)[0]
            log_prob = logistic_model.predict_proba(input_data)[0][1]  # Probability of disease (class 1)
            
            rf_pred = random_forest_model.predict(input_data)[0]
            rf_prob = random_forest_model.predict_proba(input_data)[0][1]  # Probability of disease (class 1)
            
            # Display individual predictions
            st.markdown('<div class="subheader">Model Predictions</div>', unsafe_allow_html=True)
            st.write(f"**Logistic Regression**: {'Has Heart Disease' if log_pred == 1 else 'No Heart Disease'} (Confidence: {log_prob:.2f})")
            st.write(f"**Random Forest**: {'Has Heart Disease' if rf_pred == 1 else 'No Heart Disease'} (Confidence: {rf_prob:.2f})")
            
            # Decide the best prediction: choose the one with higher confidence probability for class 1 or 0
            if log_prob > rf_prob:
                best_pred = log_pred
                best_model = "Logistic Regression"
                best_conf = log_prob
            else:
                best_pred = rf_pred
                best_model = "Random Forest"
                best_conf = rf_prob
            
            # If probabilities are for no disease, compare 1 - prob
            if best_pred == 0:
                log_conf_no = 1 - log_prob
                rf_conf_no = 1 - rf_prob
                if log_conf_no > rf_conf_no:
                    best_pred = log_pred
                    best_model = "Logistic Regression"
                    best_conf = log_conf_no
                else:
                    best_pred = rf_pred
                    best_model = "Random Forest"
                    best_conf = rf_conf_no
            
            # Final output
            st.markdown('<div class="subheader">Final Prediction</div>', unsafe_allow_html=True)
            if best_pred == 1:
                st.error(f"The patient is predicted to have heart disease based on {best_model} (Confidence: {best_conf:.2f}).")
            else:
                st.success(f"The patient is predicted to NOT have heart disease based on {best_model} (Confidence: {best_conf:.2f}).")
