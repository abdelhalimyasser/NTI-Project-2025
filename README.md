 # ❤️ UCI Heart Disease Prediction  

This repository contains my **Machine Learning project** for the **NTI ML Training Program**, focused on predicting heart disease using the **UCI Heart Disease Dataset**. 

The goal is to apply data preprocessing, exploratory analysis, and machine learning models to classify patients based on the likelihood of having heart disease.  

---

## 📊 Dataset  
- **Source:** [UCI Machine Learning Repository – Heart Disease Dataset - from Kaggle](https://www.kaggle.com/datasets/mragpavank/heart-diseaseuci)  
- **Features:** Age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, ECG results, max heart rate, exercise-induced angina, ST depression, slope, vessels, thalassemia.  
- **Target:** Presence (1) or absence (0) of heart disease.  

---

## ⚙️ Project Workflow  
1. **Data Preprocessing**  
   - Missing values handling
   - Remove Duplicates
   - Removing Outliers
   - Encoding categorical features  
   - Feature scaling  
   - Splitting into train/test sets  

2. **Exploratory Data Analysis (EDA)**  
   - 📈 Correlation heatmaps  
   - 📊 Distribution plots  
   - 🔎 Insights into key risk factors  

3. **Model Building**  
   - Algorithms: `Logistic Regression`, `Random Forest`, `Decision Tree`  
   - Applied **cross-validation**  

4. **Model Evaluation**  
   - Metrics: `Accuracy`, `Precision`, `Recall`, `F1-score`  
   - Compared performance of multiple models  

---

## 🚀 Try it Out  

You can interact with the live demo of this project through **Streamlit App**:  

👉 [Heart Disease Prediction – Streamlit App](https://heart-disease-prediction-nti.streamlit.app/)  

This app allows you to:  
- Input patient data (age, sex, chest pain type, cholesterol, etc.)  
- Get real-time predictions whether the patient is at risk of heart disease  
- Visualize the results and understand the model’s decision-making
- Downloading Prediction Report

<p align="center">
  <img src="https://youtu.be/YvsRaySnnL8" width="600"/>
</p>
[![Watch the demo](https://img.youtube.com/vi/YvsRaySnnL8/0.jpg)](https://youtu.be/YvsRaySnnL8)



---

## 📌 Results  
- **Random Forest** and **Logistic Regression** performed best  
- Key features influencing predictions: chest pain type, cholesterol, max heart rate  
- ML can provide valuable support in **healthcare decision-making**  

---

## 🛠️ Tech Stack  
- **Python:** `Python 3.13.7` 
- **Libraries:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-learn`  
- **Notebook:** `Google Colab` & `Jupyter`  

---

## 📑 Reports & Diagrams  
This project provides:  
- **Reports:** Model performance summaries & evaluation metrics  
- **Diagrams:**  
  - Correlation heatmap  
  - Feature distribution plots  
  - ROC-AUC curve  

(*Add screenshots of your plots here for better illustration*)  

---

## 📚 Learning Outcomes  
- Applied a **full ML pipeline** (preprocessing → training → evaluation)  
- Hands-on experience with **classification problems**  
- Stronger understanding of **EDA and feature importance**  

---

## 📝 License
This project is licensed under the [MIT License](./LICENSE).  
You are free to use, modify, and distribute the code with proper attribution.

---

© 2025 Abdelhalim Yasser – Released under the MIT License.  
