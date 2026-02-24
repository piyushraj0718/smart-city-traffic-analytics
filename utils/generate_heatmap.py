"""
Advanced Heatmap Module
Smart City Traffic & Accident Risk Analytics System

Features:
- Severity-based heatmap
- Time category filtering
- City filtering
"""

import os
import pandas as pd
import folium
from folium.plugins import HeatMap


# Configuration

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "ml_ready_accidents.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "heatmap")

os.makedirs(OUTPUT_DIR, exist_ok=True)



# Load Data

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["Latitude", "Longitude"])
df = df[(df["Latitude"] != 0) & (df["Longitude"] != 0)]

print(f"Total records after cleaning: {len(df)}")



# 🔵 Filter Controls (Change Here Manually for Now)


FILTER_SEVERITY = "None"   # Options: "Low", "Medium", "High", or None
FILTER_TIME = None       # Options: "Morning", "Afternoon", "Evening", "Night", or None
FILTER_CITY = None       # Example: "London", "Birmingham", etc. or None


# Apply severity filter
if FILTER_SEVERITY:
    df = df[df["Severity_Label"] == FILTER_SEVERITY]
    print(f"Filtered by Severity: {FILTER_SEVERITY}")

# Apply time filter
if FILTER_TIME:
    df = df[df["Time_Category"] == FILTER_TIME]
    print(f"Filtered by Time: {FILTER_TIME}")

# Apply city filter
if FILTER_CITY and "Local_Authority_(District)" in df.columns:
    df = df[df["Local_Authority_(District)"].str.contains(FILTER_CITY, case=False)]
    print(f"Filtered by City: {FILTER_CITY}")


print(f"Records after filtering: {len(df)}")



# Sampling for performance

sample_size = min(20000, len(df))
df_sample = df.sample(sample_size, random_state=42)

print(f"Using {sample_size} points for visualization.")



# Create Map

map_center = [df_sample["Latitude"].mean(), df_sample["Longitude"].mean()]

heatmap_map = folium.Map(
    location=map_center,
    zoom_start=6,
    tiles="CartoDB positron"
)

heat_data = list(zip(df_sample["Latitude"], df_sample["Longitude"]))

HeatMap(
    heat_data,
    radius=8,
    blur=12,
    min_opacity=0.4
).add_to(heatmap_map)



# Save Output

heatmap_path = os.path.join(OUTPUT_DIR, "advanced_accident_heatmap.html")
heatmap_map.save(heatmap_path)

print(f"Advanced heatmap saved at: {heatmap_path}")
print("Advanced heatmap generation completed.")