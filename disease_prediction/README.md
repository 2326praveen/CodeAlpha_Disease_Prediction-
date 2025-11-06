# Heart Disease Prediction (Streamlit)

This workspace contains a Streamlit app `disease_prediction_app.py` that predicts heart disease from medical features, and a training script `train_and_save.py` to produce the saved model artifacts.

Files:
- `disease_prediction_app.py` - Streamlit app. Reads `heart.csv` and expects `best_model.pkl` and `scaler.pkl` to be present (or use the Train Models button in the UI).
- `train_and_save.py` - Run this to train models and save `best_model.pkl` and `scaler.pkl` without using the UI.
- `heart.csv` - dataset (already present at workspace root).

Quick start (PowerShell on Windows):

1) Create a venv and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Train and save model artifacts (optional -- the Streamlit UI can also train):

```powershell
python train_and_save.py
```

3) Launch the Streamlit app:

```powershell
streamlit run disease_prediction_app.py
```

Notes:
- If you don't want to run `train_and_save.py`, open the app in the browser and click "Train Models" to train from the UI. That will also save `best_model.pkl` and `scaler.pkl` for subsequent runs.
- If `xgboost` is not available, `train_and_save.py` will fail; you can remove XGBoost from the models dict or install it via `pip install xgboost`.
