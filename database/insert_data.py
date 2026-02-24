import pandas as pd
import os
from db_connection import insert_dataframe

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "ml_ready_accidents.csv")


def main():
    print("Loading ML-ready dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset=["Accident_Index"])
    # Select only relevant columns matching schema
    df = df[[
        "Accident_Index",
        "Accident_Severity",
        "Severity_Label",
        "Number_of_Vehicles",
        "Number_of_Casualties",
        "Date",
        "Time",
        "Day_of_Week",
        "Latitude",
        "Longitude",
        "Weather_Conditions",
        "Road_Type",
        "Light_Conditions",
        "Road_Surface_Conditions",
        "Time_Category",
        "Is_Weekend",
        "Weather_Severity_Index",
        "Road_Risk_Score"
    ]]

    # Rename to match SQL schema
    df.columns = [
        "accident_index",
        "accident_severity",
        "severity_label",
        "number_of_vehicles",
        "number_of_casualties",
        "date",
        "time",
        "day_of_week",
        "latitude",
        "longitude",
        "weather_conditions",
        "road_type",
        "light_conditions",
        "road_surface_conditions",
        "time_category",
        "is_weekend",
        "weather_severity_index",
        "road_risk_score"
    ]

    insert_dataframe(df)


if __name__ == "__main__":
    main()