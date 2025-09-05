import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the pre-trained models
try:
    logistic_model = joblib.load("logistic_model.pkl")
    random_forest_model = joblib.load("random_forest_model.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'logistic_model.pkl' and 'random_forest_model.pkl' are in the same directory.")
    st.stop()

# Streamlit app title
st.title("Heart Disease Prediction App")

# Sidebar for user inputs
st.sidebar.header("Enter Patient Details")

# Input fields for features
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=50)
sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.sidebar.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure (trestbps)", min_value=0, max_value=300, value=120)
chol = st.sidebar.number_input("Serum Cholestoral (chol)", min_value=0, max_value=600, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
restecg = st.sidebar.selectbox("Resting ECG Results (restecg)", options=[0, 1, 2])
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved (thalach)", min_value=0, max_value=250, value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", options=[0, 1])
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (slope)", options=[0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", options=[0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

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

# Predict button
if st.sidebar.button("Predict"):
    # Predictions
    log_pred = logistic_model.predict(input_data)[0]
    log_prob = logistic_model.predict_proba(input_data)[0][1]  # Probability of disease (class 1)
    
    rf_pred = random_forest_model.predict(input_data)[0]
    rf_prob = random_forest_model.predict_proba(input_data)[0][1]  # Probability of disease (class 1)
    
    # Display individual predictions
    st.subheader("Predictions from Models:")
    st.write(f"Logistic Regression Prediction: {'Has Heart Disease' if log_pred == 1 else 'No Heart Disease'} (Confidence: {log_prob:.2f})")
    st.write(f"Random Forest Prediction: {'Has Heart Disease' if rf_pred == 1 else 'No Heart Disease'} (Confidence: {rf_prob:.2f})")
    
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
    st.subheader("Final Prediction:")
    if best_pred == 1:
        st.error(f"The patient is predicted to have heart disease based on {best_model} (Confidence: {best_conf:.2f}).")
    else:
        st.success(f"The patient is predicted to NOT have heart disease based on {best_model} (Confidence: {best_conf:.2f}).")
