"""
Machine Learning Training Pipeline (Improved with SMOTE)
Smart City Traffic & Accident Risk Analytics System

Trains:
- Logistic Regression
- Random Forest
- XGBoost

Handles:
- Class imbalance (SMOTE)
- Model comparison
- Evaluation metrics
- Model saving
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE



# Safe Path Handling

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "ml_ready_accidents.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "trained_models")

os.makedirs(MODEL_DIR, exist_ok=True)



# Load Data

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("Dataset loaded.")



# Feature Selection

features = [
    "Number_of_Vehicles",
    "Number_of_Casualties",
    "Speed_limit",
    "Weather_Severity_Index",
    "Road_Risk_Score",
    "Is_Weekend"
]

# Keep only existing columns
features = [col for col in features if col in df.columns]

X = df[features]
y = df["Severity_Label"]



# Encode Target

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)



# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)



# Handle Class Imbalance using SMOTE

print("Applying SMOTE for class balancing...")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("Original class distribution:", np.bincount(y_train))
print("Balanced class distribution:", np.bincount(y_train_balanced))



# Scale Features (Logistic Regression only)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)



# 1️ Logistic Regression

print("\nTraining Logistic Regression...")

log_model = LogisticRegression(
    max_iter=1000
)

log_model.fit(X_train_scaled, y_train_balanced)
log_preds = log_model.predict(X_test_scaled)



# 2️ Random Forest

print("Training Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train_balanced, y_train_balanced)
rf_preds = rf_model.predict(X_test)



# 3️ XGBoost

print("Training XGBoost...")

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="mlogloss"
)

xgb_model.fit(X_train_balanced, y_train_balanced)
xgb_preds = xgb_model.predict(X_test)



# Evaluation Function

def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Results")
    print("-" * 50)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score (weighted):", f1_score(y_true, y_pred, average="weighted"))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


evaluate_model("Logistic Regression", y_test, log_preds)
evaluate_model("Random Forest", y_test, rf_preds)
evaluate_model("XGBoost", y_test, xgb_preds)



# Save Best Model (Using XGBoost now)

print("\nSaving best model...")

joblib.dump(xgb_model, os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("Model saved successfully.")