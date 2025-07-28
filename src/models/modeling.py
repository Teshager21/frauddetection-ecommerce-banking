# task_2_modeling.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

# import matplotlib.pyplot as plt
# import seaborn as sns
import joblib


# --- 0. Encoding Helper: Frequency encoding ---
def frequency_encode(train_df, test_df, column):
    freq = train_df[column].value_counts(normalize=True)
    train_df[f"{column}_freq"] = train_df[column].map(freq)
    test_df[f"{column}_freq"] = test_df[column].map(freq).fillna(0)  # unseen values → 0
    train_df.drop(columns=[column], inplace=True)
    test_df.drop(columns=[column], inplace=True)
    return train_df, test_df


# --- Helper: Evaluate model ---
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {name} ===")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
    print("Average Precision (AUC-PR):", average_precision_score(y_test, y_prob))


# --- Helper: Train-test split with stratification ---
def split_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# --- 1. Load preprocessed data ---
df_fraud = pd.read_csv("data/interim/interim_fraud_data.csv")
df_credit = pd.read_csv("data/interim/interim_creditcard.csv")

# --- 2. Split each dataset ---
X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = split_data(
    df_fraud, "is_fraud"
)
X_credit_train, X_credit_test, y_credit_train, y_credit_test = split_data(
    df_credit, "Class"
)

# --- 3. Frequency Encode device_id (or any categorical columns) ---
# Apply only to the fraud dataset (adjust column name if needed)
# if "device_id" in X_fraud_train.columns:
#     X_fraud_train, X_fraud_test =
# frequency_encode(X_fraud_train, X_fraud_test, "device_id")

# --- 4. Logistic Regression Pipelines ---
pipe_fraud_log = LogisticRegression(class_weight="balanced", max_iter=1000)
pipe_credit_log = LogisticRegression(class_weight="balanced", max_iter=1000)

# For the fraud (e-commerce) dataset
pipe_fraud_log = make_pipeline(StandardScaler(), LogisticRegression())
pipe_fraud_log.fit(X_fraud_train, y_fraud_train)
evaluate_model(
    "Logistic Regression (Fraud)", pipe_fraud_log, X_fraud_test, y_fraud_test
)

# For the banking dataset
pipe_credit_log = make_pipeline(StandardScaler(), LogisticRegression())
pipe_credit_log.fit(X_credit_train, y_credit_train)
evaluate_model(
    "Logistic Regression (Bank)", pipe_credit_log, X_credit_test, y_credit_test
)

# pipe_fraud_log.fit(X_fraud_train, y_fraud_train)xgb_credit
# pipe_credit_log.fit(X_credit_train, y_credit_train)


# --- 5. XGBoost Classifier ---
def compute_scale_pos_weight(y):
    return np.sum(y == 0) / np.sum(y == 1)


xgb_fraud = XGBClassifier(
    scale_pos_weight=compute_scale_pos_weight(y_fraud_train),
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)

xgb_credit = XGBClassifier(
    scale_pos_weight=compute_scale_pos_weight(y_credit_train),
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)

xgb_fraud.fit(X_fraud_train, y_fraud_train)
xgb_credit.fit(X_credit_train, y_credit_train)

# --- 6. Evaluate Models ---
print("\n--- EVALUATION: E-Commerce Fraud Dataset ---")
evaluate_model(
    "Logistic Regression (Fraud)", pipe_fraud_log, X_fraud_test, y_fraud_test
)
evaluate_model("XGBoost (Fraud)", xgb_fraud, X_fraud_test, y_fraud_test)

print("\n--- EVALUATION: Bank Credit Transaction Dataset ---")
evaluate_model(
    "Logistic Regression (CreditCard)", pipe_credit_log, X_credit_test, y_credit_test
)
evaluate_model("XGBoost (CreditCard)", xgb_credit, X_credit_test, y_credit_test)

# --- 7. Save models ---
joblib.dump(pipe_fraud_log, "models/log_fraud.pkl")
joblib.dump(xgb_fraud, "models/xgb_fraud.pkl")
joblib.dump(pipe_credit_log, "models/log_credit.pkl")
joblib.dump(xgb_credit, "models/xgb_credit.pkl")
