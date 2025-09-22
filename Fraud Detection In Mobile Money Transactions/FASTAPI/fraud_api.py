# file: fraud_api.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import List

# ============================
# Define request schema (raw fields)
# ============================
class Transaction(BaseModel):
    transaction_id: str
    user_id: str
    transaction_type: str
    amount: float
    location: str
    device_type: str
    network_provider: str
    user_type: str
    is_foreign_number: int
    is_sim_recently_swapped: int
    has_multiple_accounts: int
    datetime: str  # ISO format: "2025-08-25 14:30:00"
    hour: int
    day_of_week: int
    time_of_day: str


# ============================
# Load trained models
# ============================
iso_pipe = joblib.load("iso_pipeline.pkl")
ocsvm_pipe = joblib.load("ocsvm_pipeline.pkl")
lof_pipe = joblib.load("lof_pipeline.pkl")

app = FastAPI(title="Fraud Detection API")

# ============================
# In-memory transaction history (for feature engineering)
# ============================
history: List[dict] = []


# ============================
# Feature Engineering (same as notebook version)
# ============================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Transaction Frequency per User
    user_tx_count = df.groupby('user_id')['transaction_id'].count().rename('user_transaction_count')
    df = df.merge(user_tx_count, on='user_id', how='left')

    # 2. Average Transaction Amount per User
    user_avg_amount = df.groupby('user_id')['amount'].mean().rename('user_avg_amount')
    df = df.merge(user_avg_amount, on='user_id', how='left')

    # 3. Total Amount per User
    user_total_amount = df.groupby('user_id')['amount'].sum().rename('user_total_amount')
    df = df.merge(user_total_amount, on='user_id', how='left')

    # 4. Amount Deviation from User Average
    df['amount_dev_from_avg'] = df['amount'] - df['user_avg_amount']

    # 5. Risk Score (sum of binary fraud flags)
    df['risk_score'] = df[['is_foreign_number', 'is_sim_recently_swapped', 'has_multiple_accounts']].sum(axis=1)

    # 6. High Value Flag (based on 95th percentile of current history)
    high_value_threshold = df['amount'].quantile(0.95)
    df['is_high_value'] = (df['amount'] > high_value_threshold).astype(int)

    return df


# ============================
# Predict endpoint (POST)
# ============================
@app.post("/predict")
def predict(transaction: Transaction):
    global history

    # Append new transaction to history
    history.append(transaction.dict())

    # Convert history into dataframe
    df = pd.DataFrame(history)

    # Run feature engineering on full history
    df = feature_engineering(df)

    # Take the most recent transaction only
    tx = df.iloc[[-1]]

    # Run through each model
    iso_score = iso_pipe.decision_function(tx)
    iso_flag = (iso_score < 0).astype(int)

    svm_score = ocsvm_pipe.decision_function(tx)
    svm_flag = (svm_score < 0).astype(int)

    lof_score = lof_pipe.score_samples(tx)
    lof_flag = (lof_score < np.quantile(lof_score, 0.02)).astype(int)

    # Ensemble (majority vote)
    flags = np.vstack([iso_flag, svm_flag, lof_flag])
    ensemble_flag = (flags.sum(axis=0) >= 2).astype(int)[0]

    return {
        "isolation_forest": {"score": float(iso_score[0]), "flag": int(iso_flag[0])},
        "oneclass_svm": {"score": float(svm_score[0]), "flag": int(svm_flag[0])},
        "lof": {"score": float(lof_score[0]), "flag": int(lof_flag[0])},
        "ensemble_majority": int(ensemble_flag)
    }


# ============================
# Simple test endpoint (GET)
# ============================
@app.get("/predict")
def test_predict():
    return {"message": "Fraud API is running. Use POST /predict with JSON transaction data."}
