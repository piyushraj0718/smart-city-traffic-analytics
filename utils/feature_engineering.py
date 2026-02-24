"""
Feature Engineering Module
Smart City Traffic Analytics System

Creates intelligent features for ML modeling.
"""

import pandas as pd
import numpy as np
import os

CLEAN_DATA_PATH = "data/processed/cleaned_accidents.csv"
ML_READY_PATH = "data/processed/ml_ready_accidents.csv"


def load_clean_data():
    print("Loading cleaned dataset...")
    return pd.read_csv(CLEAN_DATA_PATH)



# Time Category Feature

def create_time_category(df):
    print("Creating time category feature...")

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    df["Hour"] = df["Time"].dt.hour

    def map_time(hour):
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"

    df["Time_Category"] = df["Hour"].apply(map_time)

    return df


# Weekend Flag

def create_weekend_flag(df):
    print("Creating weekend flag...")

    # If Day_of_Week column exists (1=Sunday, 7=Saturday typical UK format)
    if "Day_of_Week" in df.columns:
        df["Is_Weekend"] = df["Day_of_Week"].isin([1, 7]).astype(int)
    else:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Is_Weekend"] = df["Date"].dt.dayofweek.isin([5, 6]).astype(int)

    return df



# 3️ Weather Severity Index

def create_weather_severity_index(df):
    print("Creating weather severity index...")

    weather_risk_mapping = {
        "Fine without high winds": 1,
        "Fine with high winds": 2,
        "Raining without high winds": 3,
        "Raining with high winds": 4,
        "Snowing without high winds": 4,
        "Snowing with high winds": 5,
        "Fog or mist": 3,
        "Other": 2,
        "Unknown": 2
    }

    if "Weather_Conditions" in df.columns:
        df["Weather_Severity_Index"] = df["Weather_Conditions"].map(weather_risk_mapping)
        df["Weather_Severity_Index"] = df["Weather_Severity_Index"].fillna(2)
    else:
        df["Weather_Severity_Index"] = 2

    return df



# 4️ Road Risk Score

def create_road_risk_score(df):
    print("Creating road risk score...")

    road_risk_mapping = {
        "Single carriageway": 3,
        "Dual carriageway": 2,
        "Roundabout": 1,
        "One way street": 2,
        "Slip road": 3,
        "Unknown": 2
    }

    if "Road_Type" in df.columns:
        df["Road_Risk_Score"] = df["Road_Type"].map(road_risk_mapping)
        df["Road_Risk_Score"] = df["Road_Risk_Score"].fillna(2)
    else:
        df["Road_Risk_Score"] = 2

    return df



# 5️ Severity Label Mapping

def map_severity_label(df):
    print("Mapping severity labels...")

    severity_mapping = {
        1: "High",     # Fatal
        2: "Medium",   # Serious
        3: "Low"       # Slight
    }

    df["Severity_Label"] = df["Accident_Severity"].map(severity_mapping)

    return df



# Save ML Ready Dataset

def save_ml_ready_data(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(ML_READY_PATH, index=False)
    print("ML-ready dataset saved successfully.")


def run_feature_engineering_pipeline():
    df = load_clean_data()
    df = create_time_category(df)
    df = create_weekend_flag(df)
    df = create_weather_severity_index(df)
    df = create_road_risk_score(df)
    df = map_severity_label(df)
    save_ml_ready_data(df)
    print("Feature engineering completed successfully.")
    return df


if __name__ == "__main__":
    run_feature_engineering_pipeline()