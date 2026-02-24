"""
Accident Forecasting Module
Smart City Traffic & Accident Risk Analytics System

Forecasts daily accident trends using Prophet.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet



# Path Setup

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "ml_ready_accidents.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "forecast")

os.makedirs(OUTPUT_DIR, exist_ok=True)



# Load Dataset

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])



# Aggregate Daily Accident Counts

print("Aggregating daily accident counts...")

daily_accidents = df.groupby("Date").size().reset_index()
daily_accidents.columns = ["ds", "y"]  # Prophet format


print(f"Total days available: {len(daily_accidents)}")



# Train Prophet Model

print("Training Prophet model...")

model = Prophet(
    daily_seasonality=True,
    yearly_seasonality=True,
    weekly_seasonality=True
)

model.fit(daily_accidents)



# Create Future Dates

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)



# Save Forecast Data

forecast_path = os.path.join(OUTPUT_DIR, "30_day_forecast.csv")
forecast.to_csv(forecast_path, index=False)

print(f"Forecast data saved at: {forecast_path}")



# Plot Forecast

print("Generating forecast plot...")

fig = model.plot(forecast)
plt.title("Accident Forecast (Next 30 Days)")

plot_path = os.path.join(OUTPUT_DIR, "accident_forecast.png")
plt.savefig(plot_path, bbox_inches="tight")
plt.close()

print(f"Forecast plot saved at: {plot_path}")

print("Forecasting completed successfully.")