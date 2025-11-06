"""
Train models (LogisticRegression, SVC, RandomForest, XGBoost) on `heart.csv`,
select the best by accuracy on the test set, and save the model + scaler as
`best_model.pkl` and `scaler.pkl` for the Streamlit app to load.

Run:
    python train_and_save.py

This script is independent from Streamlit and intended to be run once to
produce the saved artifacts used by `disease_prediction_app.py`.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# import xgboost only if available (training will skip it otherwise)
import importlib
_spec = importlib.util.find_spec("xgboost")
if _spec is not None:
    xgb = importlib.import_module("xgboost")
    XGBClassifier = getattr(xgb, "XGBClassifier")
    XGBOOST_AVAILABLE = True
else:
    XGBOOST_AVAILABLE = False
import joblib


def main():
    print("Loading data from heart.csv...")
    df = pd.read_csv('heart.csv')

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel='rbf', probability=True),
        "Random Forest": RandomForestClassifier(random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        results[name] = {"model": model, "accuracy": acc, "f1": f1}
        print(f"  {name} -> accuracy: {acc:.4f}, f1: {f1:.4f}")

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_model = results[best_name]["model"]
    best_acc = results[best_name]["accuracy"]

    print(f"Best model: {best_name} (accuracy={best_acc:.4f})")

    print("Saving best_model.pkl and scaler.pkl...")
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print("Done.")


if __name__ == '__main__':
    main()
