"""
model.py  -  Customer Churn Prediction
Trains a Random Forest and a Logistic Regression classifier on the synthetic
telecom dataset, prints evaluation metrics, and saves the best model to model.pkl.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

# ── 1. Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/customer_churn.csv")

# ── 2. Feature engineering ───────────────────────────────────────────────────
# Encode binary / ordinal columns
le = LabelEncoder()
df["Gender_enc"]       = le.fit_transform(df["Gender"])          # Male=1 Female=0
df["Contract_enc"]     = le.fit_transform(df["ContractType"])    # ordinal-ish
df["TechSupport_enc"]  = le.fit_transform(df["TechSupport"])     # Yes=1 No=0
df["Churn_enc"]        = (df["Churn"] == "Yes").astype(int)

FEATURES = [
    "Gender_enc",
    "SeniorCitizen",
    "Tenure_Months",
    "MonthlyCharges",
    "TotalCharges",
    "Contract_enc",
    "TechSupport_enc",
]

X = df[FEATURES].values
y = df["Churn_enc"].values

# ── 3. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── 4. Define models ─────────────────────────────────────────────────────────
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    ),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ]),
}

# ── 5. Train, evaluate, select best ──────────────────────────────────────────
print("\n=== Model Evaluation ===\n")
best_model   = None
best_f1      = -1
best_name    = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"Model         : {name}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Precision   : {prec:.4f}")
    print(f"  Recall      : {rec:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  Confusion M : TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
    print()

    if f1 > best_f1:
        best_f1    = f1
        best_model = model
        best_name  = name

print(f"Best model: {best_name}  (F1 = {best_f1:.4f})")

# ── 6. Persist artefacts ─────────────────────────────────────────────────────
artefact = {
    "model":    best_model,
    "features": FEATURES,
    "model_name": best_name,
}

with open("model.pkl", "wb") as f:
    pickle.dump(artefact, f)

print("Model saved -> model.pkl")
