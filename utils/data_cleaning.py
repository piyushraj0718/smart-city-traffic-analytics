"""
Data Cleaning Module
Smart City Traffic Analytics System

Handles:
- Loading raw dataset
- Fixing date & time
- Handling missing values
- Removing invalid records
- Saving cleaned dataset
"""

import pandas as pd
import numpy as np
import os


RAW_DATA_PATH = "data/raw/UK_Accident.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_accidents.csv"


def load_raw_data():
    """Load raw CSV dataset"""
    print("Loading raw dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def clean_basic_issues(df):
    """Basic cleaning steps"""

    print("Cleaning column names...")
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # Fix Date column
    print("Parsing Date column...")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Remove rows where Date is missing
    df = df.dropna(subset=["Date"])

    # Fix Time column
    print("Parsing Time column...")
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.time

    # Drop rows with invalid time
    df = df.dropna(subset=["Time"])

    return df


def handle_missing_values(df):
    """Handle missing values intelligently"""

    print("Handling missing values...")

    # Numerical columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Categorical columns
    categorical_cols = df.select_dtypes(include="object").columns
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    return df


def remove_invalid_coordinates(df):
    """Remove records with invalid lat/long"""

    print("Removing invalid coordinates...")

    df = df[
        (df["Latitude"] != 0) &
        (df["Longitude"] != 0)
    ]

    return df


def save_cleaned_data(df):
    """Save cleaned dataset"""
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Cleaned dataset saved successfully.")


def run_cleaning_pipeline():
    """Full cleaning pipeline"""
    df = load_raw_data()
    df = clean_basic_issues(df)
    df = handle_missing_values(df)
    df = remove_invalid_coordinates(df)
    save_cleaned_data(df)
    print("Data cleaning completed successfully.")
    return df


if __name__ == "__main__":
    run_cleaning_pipeline()