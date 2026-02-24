# Smart City Traffic & Accident Risk Analytics System

## 1. Project Overview

This project is an end-to-end data analytics and machine learning system designed to analyze road accident patterns and predict accident severity.

The goal of this system is to simulate a smart city monitoring platform that can:

* Analyze historical accident trends

* Predict accident severity using machine learning

* Identify high-risk conditions

* Visualize geographic risk patterns

* Forecast future accident trends

The system integrates data engineering, machine learning, explainability techniques, and an interactive dashboard into one unified application.

## 2. Problem Statement

Road accidents remain a critical issue in urban environments. Understanding accident patterns and predicting severity can help authorities:

* Improve road safety policies

* Identify high-risk zones

* Allocate emergency resources effectively

* Make data-driven infrastructure decisions

This project aims to build a predictive and analytical framework using historical UK accident data.

## 3. Dataset

Dataset Used: UK Road Accident Dataset (2005–2014)

NOTE :- Raw dataset not included due to size. Available from UK Government dataset portal.

After cleaning and preprocessing:

* ~368,000 records

* Structured accident-level data

* Includes environmental, road, and temporal features

* Key features include:

* Accident Severity

* Number of Vehicles

* Number of Casualties

* Speed Limit

* Weather Conditions

* Road Surface Conditions

* Light Conditions

* Latitude & Longitude

* Date & Time

## Dataset Note (Deployment Version)

The original dataset used for training contains approximately 368,000 cleaned accident records (UK Road Accident Dataset 2005–2014).

Due to GitHub file size limitations, the full processed dataset is not included in this repository.

For demonstration and deployment purposes, a representative sample dataset is included in:

data/processed/sample_accidents.csv

The trained ML model, SHAP analysis, and forecasting modules were built using the complete dataset.

## 4. System Architecture

The project follows a modular structure:

* Data Cleaning & Preprocessing

* Feature Engineering

* Model Training & Evaluation

* Explainability Module (SHAP)

* Time-Series Forecasting (Prophet)

* Geospatial Heatmap Generation

* Streamlit Dashboard Integration

The system is organized into separate modules for maintainability and scalability.

## 5. Data Processing & Feature Engineering

Steps performed:

* Handling missing values

* Encoding categorical variables

* Creating derived features such as:

* Weather Severity Index

* Road Risk Score

* Weekend Indicator

* Aggregating daily accident counts for forecasting

The final ML-ready dataset was stored for consistent model input.

## 6. Machine Learning Pipeline
Models Evaluated:

* Logistic Regression

* Random Forest

* XGBoost

To address class imbalance, SMOTE (Synthetic Minority Oversampling Technique) was applied.

Final Model Selected:

# XGBoost (based on overall classification performance)

Model predicts accident severity using:

* Number_of_Vehicles

* Number_of_Casualties

* Speed_limit

* Weather_Severity_Index

* Road_Risk_Score

* Is_Weekend

The trained model is serialized and integrated into the dashboard for real-time predictions.

## 7. Model Explainability

To ensure interpretability, SHAP (SHapley Additive exPlanations) was implemented.

Generated outputs:

* SHAP Beeswarm Plot

* SHAP Feature Importance Bar Plot

This helps understand which features most influence severity predictions.

## 8. Forecasting Module

A time-series forecasting model was built using Facebook Prophet.

* Daily accident counts aggregated

* 30-day future forecast generated

* Confidence intervals visualized

This simulates predictive city monitoring capability.

## 9. Geospatial Risk Visualization

A Folium-based heatmap was developed to visualize:

* Accident density

* Geographic risk distribution

This enables spatial pattern analysis across the UK.

## 10. Interactive Dashboard

Built using Streamlit with a modular UI.

Dashboard Sections:

* Executive Overview

* ML Prediction Engine

* Risk Heatmap

* 30-Day Forecast

The interface is designed to simulate a city-level accident intelligence monitoring platform.

## 11. Tech Stack

* Python

* Pandas & NumPy

* Scikit-learn

* XGBoost

* SHAP

* Prophet

* Folium

* Streamlit

## 12. Key Learnings

Through this project, I gained practical experience in:

* Handling large real-world datasets

* Dealing with class imbalance

* Model comparison and evaluation

* Explainable AI techniques

* Time-series forecasting

* End-to-end system integration

* Building interactive analytical dashboards

## 13. Future Improvements

* Real-time accident data integration

* API deployment for prediction service

* Advanced spatial clustering

* Model monitoring & performance tracking