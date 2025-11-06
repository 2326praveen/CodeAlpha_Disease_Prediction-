# disease_prediction_app.py

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# import xgboost if available using importlib to avoid raising at parse time
import importlib
_spec = importlib.util.find_spec("xgboost")
if _spec is not None:
    xgb = importlib.import_module("xgboost")
    XGBClassifier = getattr(xgb, "XGBClassifier")
    XGBOOST_AVAILABLE = True
else:
    XGBOOST_AVAILABLE = False
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# LOAD AND PREPROCESS DATA
# -------------------------
@st.cache_data
def load_data():
    # the dataset is located at the workspace root as `heart.csv`
    df = pd.read_csv('heart.csv')
    return df

df = load_data()

st.title("💖 Heart Disease Prediction System")
st.write("Predict whether a person has heart disease based on medical attributes.")

# Show dataset preview
if st.checkbox("Show Dataset"):
    st.dataframe(df.head())

# -------------------------
# FEATURE SELECTION & SPLIT
# -------------------------
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------
# FEATURE SCALING
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------
# MODEL TRAINING FUNCTION
# -------------------------
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel='rbf', probability=True),
        "Random Forest": RandomForestClassifier(random_state=42),
    }
    if XGBOOST_AVAILABLE:
        # configure XGBoost only when available
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        results[name] = {"model": model, "accuracy": acc, "f1": f1}
    
    return results

# -------------------------
# TRAIN MODELS
# -------------------------
if st.button("Train Models"):
    results = train_models(X_train, X_test, y_train, y_test)
    result_df = pd.DataFrame({name: [r["accuracy"], r["f1"]] for name, r in results.items()},
                             index=["Accuracy", "F1 Score"]).T
    st.subheader("📊 Model Comparison")
    st.dataframe(result_df)

    best_model_name = max(results, key=lambda x: results[x]["accuracy"])
    best_model = results[best_model_name]["model"]
    st.success(f"✅ Best Model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.3f})")

    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    st.info("Model and Scaler saved successfully!")

# -------------------------
# MODEL EVALUATION VISUALIZATION
# -------------------------
if st.checkbox("Show Correlation Heatmap"):
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# -------------------------
# PREDICTION SECTION
# -------------------------
st.header("🧍 Predict Heart Disease")

# Load saved model if available
try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.warning("⚠️ Please train the model first.")
    st.stop()

# Collect input data
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 20, 100, 45)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [1, 0])
with col2:
    restecg = st.number_input("Rest ECG (0-2)", 0, 2, 1)
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [1, 0])
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
    slope = st.number_input("Slope (0-2)", 0, 2, 1)
    ca = st.number_input("Major Vessels (0-4)", 0, 4, 0)
    thal = st.number_input("Thal (0=Normal, 1=Fixed Defect, 2=Reversible Defect)", 0, 2, 1)

# Make prediction
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The model predicts a **High Risk** of Heart Disease (Probability: {probability:.2f})")
    else:
        st.success(f"💚 The model predicts **Low Risk** of Heart Disease (Probability: {probability:.2f})")
