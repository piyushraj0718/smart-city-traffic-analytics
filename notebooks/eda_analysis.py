"""
Exploratory Data Analysis
Smart City Traffic & Accident Risk Analytics System

Generates:
- Year trend
- Hour distribution
- Severity distribution
- Weather impact
- Road type impact
- Correlation heatmap
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Safe Path Handling (Production Ready)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "ml_ready_accidents.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)



# Load Dataset

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")



# Convert Date & Time Properly

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year

# If Hour column does not exist, recreate it safely
if "Hour" not in df.columns:
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["Hour"] = df["Time"].dt.hour


# 1️ Accident Trend by Year

plt.figure(figsize=(10, 6))
yearly_trend = df.groupby("Year").size()

yearly_trend.plot(marker="o")
plt.title("Accident Trend by Year")
plt.ylabel("Number of Accidents")
plt.xlabel("Year")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "accident_trend_by_year.png"))
plt.close()



# 2️ Accident by Hour

plt.figure(figsize=(10, 6))
hourly = df.groupby("Hour").size()

hourly.plot()
plt.title("Accidents by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Number of Accidents")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "accidents_by_hour.png"))
plt.close()



# 3️⃣ Severity Distribution

plt.figure(figsize=(8, 6))
severity_counts = df["Severity_Label"].value_counts()

sns.barplot(x=severity_counts.index, y=severity_counts.values)
plt.title("Accident Severity Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "severity_distribution.png"))
plt.close()



# 4️ Weather Impact on Severity

plt.figure(figsize=(12, 6))
weather_severity = pd.crosstab(
    df["Weather_Conditions"],
    df["Severity_Label"]
)

weather_severity.plot(kind="bar", stacked=True)
plt.title("Weather Impact on Severity")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "weather_impact.png"))
plt.close()



# 5️ Road Type Impact

plt.figure(figsize=(12, 6))
road_severity = pd.crosstab(
    df["Road_Type"],
    df["Severity_Label"]
)

road_severity.plot(kind="bar", stacked=True)
plt.title("Road Type Impact on Severity")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "road_type_impact.png"))
plt.close()



# 6️⃣ Correlation Heatmap

plt.figure(figsize=(10, 8))

numeric_cols = df.select_dtypes(include="number")
corr = numeric_cols.corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "correlation_heatmap.png"))
plt.close()


print("EDA completed successfully. All plots saved in reports folder.")