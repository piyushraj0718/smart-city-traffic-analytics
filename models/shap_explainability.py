"""
SHAP Explainability Module
Smart City Traffic & Accident Risk Analytics System

Generates SHAP plots for trained XGBoost model.
Compatible with XGBoost 2.x and multi-class models.
"""

import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt



# Path Configuration

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_models", "best_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "ml_ready_accidents.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "shap")

os.makedirs(OUTPUT_DIR, exist_ok=True)



# Load Model

print("Loading trained model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")



# Load Dataset

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

features = [
    "Number_of_Vehicles",
    "Number_of_Casualties",
    "Speed_limit",
    "Weather_Severity_Index",
    "Road_Risk_Score",
    "Is_Weekend"
]

features = [col for col in features if col in df.columns]

X = df[features]

print(f"Dataset ready. Using {len(features)} features.")



# Sampling (safe memory handling)

sample_size = min(5000, len(X))
sample_X = X.sample(sample_size, random_state=42)

print(f"Using {sample_size} samples for SHAP.")



# Create SHAP Explainer (Native Booster Mode)

print("Initializing SHAP explainer (Booster mode)...")

booster = model.get_booster()
explainer = shap.TreeExplainer(booster)

print("Calculating SHAP values...")
shap_values = explainer(sample_X)



# Handle Multi-class Output

if hasattr(shap_values, "values"):
    values = shap_values.values
else:
    values = shap_values

# If multi-class → values shape: (samples, features, classes)
if len(values.shape) == 3:
    print("Multi-class model detected. Using class index 0 for visualization.")
    values = values[:, :, 0]


# SHAP Beeswarm Plot

print("Generating SHAP beeswarm plot...")

plt.figure()
shap.summary_plot(values, sample_X, show=False)

beeswarm_path = os.path.join(OUTPUT_DIR, "shap_beeswarm.png")
plt.savefig(beeswarm_path, bbox_inches="tight")
plt.close()

print(f"Beeswarm plot saved at: {beeswarm_path}")



# SHAP Bar Plot (Feature Importance)

print("Generating SHAP bar plot...")

plt.figure()
shap.summary_plot(values, sample_X, plot_type="bar", show=False)

bar_path = os.path.join(OUTPUT_DIR, "shap_bar.png")
plt.savefig(bar_path, bbox_inches="tight")
plt.close()

print(f"Bar plot saved at: {bar_path}")

print("SHAP analysis completed successfully.")